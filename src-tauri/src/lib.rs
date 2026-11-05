use std::fs::{self, OpenOptions};
use std::io::Write;
use std::net::TcpStream;
use std::path::{Path, PathBuf};
use std::process::{Child, Command};
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{Manager, WindowEvent};

// Atomic state for process supervision and graceful shutdown
static MANAGED_CHILD_PID: AtomicI32 = AtomicI32::new(0);
static IS_SHUTTING_DOWN: AtomicBool = AtomicBool::new(false);
static LOCK_FILE_FD: AtomicI32 = AtomicI32::new(-1);

static CHILD_PROCESS: Mutex<Option<Child>> = Mutex::new(None);

/// Return standard application support data and run directory for JARVIS.
fn get_app_data_dir() -> PathBuf {
    if let Ok(dir) = std::env::var("JARVIS_DATA_DIR") {
        return PathBuf::from(dir);
    }
    if let Some(home) = dirs_home() {
        home.join("Library")
            .join("Application Support")
            .join("JARVIS")
    } else {
        PathBuf::from("/tmp/jarvis")
    }
}

fn dirs_home() -> Option<PathBuf> {
    std::env::var("HOME").ok().map(PathBuf::from)
}

/// Attempt to acquire exclusive advisory file lock to enforce single instance ownership.
fn acquire_single_instance_lock() -> bool {
    let run_dir = get_app_data_dir().join("run");
    let _ = fs::create_dir_all(&run_dir);
    let lock_path = run_dir.join("jarvis.lock");
    let c_path = match std::ffi::CString::new(lock_path.to_string_lossy().as_bytes()) {
        Ok(p) => p,
        Err(_) => return false,
    };

    unsafe {
        let fd = libc::open(
            c_path.as_ptr(),
            libc::O_CREAT | libc::O_RDWR,
            0o644,
        );
        if fd < 0 {
            eprintln!("[JARVIS Desktop] Failed to open lockfile at {:?}", lock_path);
            return false;
        }

        if libc::flock(fd, libc::LOCK_EX | libc::LOCK_NB) != 0 {
            libc::close(fd);
            eprintln!(
                "[JARVIS Desktop] Another instance of JARVIS is already running. Exiting."
            );
            return false;
        }

        // Truncate and record current PID
        let _ = libc::ftruncate(fd, 0);
        let pid_str = format!("{}\n", std::process::id());
        let _ = libc::write(fd, pid_str.as_ptr() as *const libc::c_void, pid_str.len());

        LOCK_FILE_FD.store(fd, Ordering::SeqCst);
    }
    true
}

/// Release the advisory file lock cleanly.
fn release_single_instance_lock() {
    let fd = LOCK_FILE_FD.swap(-1, Ordering::SeqCst);
    if fd >= 0 {
        unsafe {
            libc::flock(fd, libc::LOCK_UN);
            libc::close(fd);
        }
    }
    let lock_path = get_app_data_dir().join("run").join("jarvis.lock");
    let _ = fs::remove_file(lock_path);
}

/// Resolve the Python interpreter executable.
fn resolve_python() -> Option<PathBuf> {
    // 1. Explicit env var
    if let Ok(py) = std::env::var("JARVIS_PYTHON") {
        let p = PathBuf::from(py);
        if p.is_file() {
            return Some(p);
        }
    }

    // 2. Relative to executable inside .app bundle
    if let Ok(exe) = std::env::current_exe() {
        if let Some(macos_dir) = exe.parent() {
            if let Some(contents_dir) = macos_dir.parent() {
                let resources_dir = contents_dir.join("Resources");
                let bundled_py = resources_dir.join("venv").join("bin").join("python");
                if bundled_py.is_file() {
                    return Some(bundled_py);
                }
            }
        }
    }

    // 3. Current working directory or project root .venv
    let cwd_venv = PathBuf::from(".venv").join("bin").join("python");
    if cwd_venv.is_file() {
        return Some(cwd_venv.canonicalize().unwrap_or(cwd_venv));
    }

    // 4. Try parent directory (if running from src-tauri)
    let parent_venv = PathBuf::from("..").join(".venv").join("bin").join("python");
    if parent_venv.is_file() {
        return Some(parent_venv.canonicalize().unwrap_or(parent_venv));
    }

    // 5. User home venv
    if let Some(home) = dirs_home() {
        let user_venv = home.join(".jarvis").join("venv").join("bin").join("python");
        if user_venv.is_file() {
            return Some(user_venv);
        }
    }

    // 6. System python3
    if let Ok(output) = Command::new("which").arg("python3").output() {
        if output.status.success() {
            let path_str = String::from_utf8_lossy(&output.stdout).trim().to_string();
            let p = PathBuf::from(path_str);
            if p.is_file() {
                return Some(p);
            }
        }
    }

    None
}

/// Locate backend entrypoint (main.py).
fn resolve_backend_entrypoint() -> Option<PathBuf> {
    // 1. Check relative to executable (.app/Contents/Resources/backend/main.py)
    if let Ok(exe) = std::env::current_exe() {
        if let Some(macos_dir) = exe.parent() {
            if let Some(contents_dir) = macos_dir.parent() {
                let bundled_main = contents_dir
                    .join("Resources")
                    .join("backend")
                    .join("main.py");
                if bundled_main.is_file() {
                    return Some(bundled_main);
                }
            }
        }
    }

    // 2. Check ./backend/main.py
    let local_main = PathBuf::from("backend").join("main.py");
    if local_main.is_file() {
        return Some(local_main.canonicalize().unwrap_or(local_main));
    }

    // 3. Check ../backend/main.py
    let parent_main = PathBuf::from("..").join("backend").join("main.py");
    if parent_main.is_file() {
        return Some(parent_main.canonicalize().unwrap_or(parent_main));
    }

    // 4. Check root ./main.py
    let root_main = PathBuf::from("main.py");
    if root_main.is_file() {
        return Some(root_main.canonicalize().unwrap_or(root_main));
    }

    None
}

/// Spawn the Python FastAPI backend bridge process as a managed child.
fn spawn_backend_service() -> Result<u32, String> {
    let python_bin = resolve_python()
        .ok_or_else(|| "Could not locate a valid Python interpreter (.venv or system python3)".to_string())?;
    let entry_script = resolve_backend_entrypoint()
        .ok_or_else(|| "Could not locate backend main.py entrypoint".to_string())?;

    let log_dir = get_app_data_dir().join("logs");
    let _ = fs::create_dir_all(&log_dir);
    let log_path = log_dir.join("backend.log");

    let log_file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&log_path)
        .map_err(|e| format!("Failed to open backend log file {:?}: {}", log_path, e))?;

    let log_file_err = log_file.try_clone()
        .map_err(|e| format!("Failed to duplicate log file descriptor: {}", e))?;

    let backend_dir = entry_script
        .parent()
        .unwrap_or_else(|| Path::new("."))
        .to_path_buf();

    println!(
        "[JARVIS Desktop] Launching backend: {:?} {:?} --bridge (cwd: {:?})",
        python_bin, entry_script, backend_dir
    );

    let child = Command::new(&python_bin)
        .arg(&entry_script)
        .arg("--bridge")
        .current_dir(&backend_dir)
        .env("PYTHONUNBUFFERED", "1")
        .env("JARVIS_DATA_DIR", get_app_data_dir().join("data"))
        .stdout(log_file)
        .stderr(log_file_err)
        .spawn()
        .map_err(|e| format!("Failed to spawn backend process: {}", e))?;

    let pid = child.id();
    MANAGED_CHILD_PID.store(pid as i32, Ordering::SeqCst);

    let mut lock = CHILD_PROCESS.lock().unwrap();
    *lock = Some(child);

    println!(
        "[JARVIS Desktop] Backend successfully spawned with PID {} (Log: {:?})",
        pid, log_path
    );
    Ok(pid)
}

/// Wait for backend bridge service to report healthy on port 8000.
fn wait_for_service_readiness(timeout_sec: u64) -> bool {
    let start = Instant::now();
    let timeout = Duration::from_secs(timeout_sec);

    while start.elapsed() < timeout {
        // Verify child has not crashed early
        let pid = MANAGED_CHILD_PID.load(Ordering::SeqCst);
        if pid > 0 {
            unsafe {
                if libc::kill(pid, 0) != 0 {
                    eprintln!("[JARVIS Desktop] Backend process exited unexpectedly during boot.");
                    return false;
                }
            }
        }

        // Test TCP connection to 127.0.0.1:8000
        if TcpStream::connect_timeout(
            &"127.0.0.1:8000".parse().unwrap(),
            Duration::from_millis(300),
        ).is_ok() {
            println!(
                "[JARVIS Desktop] Backend service online and responsive after {:.2}s.",
                start.elapsed().as_secs_f64()
            );
            return true;
        }

        std::thread::sleep(Duration::from_millis(200));
    }

    eprintln!("[JARVIS Desktop] Timed out waiting for backend to become ready.");
    false
}

/// Request graceful backend termination and stop only the owned child process.
pub fn cleanup_and_shutdown() {
    if IS_SHUTTING_DOWN.swap(true, Ordering::SeqCst) {
        return; // Idempotent guard
    }

    let pid = MANAGED_CHILD_PID.swap(0, Ordering::SeqCst);
    if pid <= 0 {
        release_single_instance_lock();
        return;
    }

    println!("[JARVIS Desktop] Initiating graceful shutdown of child PID {}...", pid);

    // 1. Request graceful HTTP shutdown via /api/shutdown endpoint
    if let Ok(mut stream) = TcpStream::connect_timeout(
        &"127.0.0.1:8000".parse().unwrap(),
        Duration::from_millis(600),
    ) {
        let req = "POST /api/shutdown HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nContent-Length: 0\r\nConnection: close\r\n\r\n";
        let _ = stream.write_all(req.as_bytes());
    }

    // 2. Send SIGTERM to the owned child PID
    unsafe {
        libc::kill(pid, libc::SIGTERM);
    }

    // 3. Allow bounded grace period (3.0 seconds) for clean shutdown
    let grace_start = Instant::now();
    let grace_limit = Duration::from_millis(3000);
    let mut still_alive = true;

    while grace_start.elapsed() < grace_limit {
        unsafe {
            if libc::kill(pid, 0) != 0 {
                still_alive = false;
                break;
            }
        }
        std::thread::sleep(Duration::from_millis(100));
    }

    // 4. Force terminate if still alive after grace period
    if still_alive {
        println!(
            "[JARVIS Desktop] Child PID {} did not terminate within grace period; sending SIGKILL...",
            pid
        );
        unsafe {
            libc::kill(pid, libc::SIGKILL);
        }
    }

    // 5. Reap the child process to prevent zombie state
    if let Ok(mut lock) = CHILD_PROCESS.lock() {
        if let Some(mut child) = lock.take() {
            let _ = child.wait();
        }
    }
    unsafe {
        let mut status = 0;
        libc::waitpid(pid, &mut status, libc::WNOHANG);
    }

    // 6. Release lockfile
    release_single_instance_lock();
    println!("[JARVIS Desktop] Shutdown and process cleanup completed.");
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    // 1. Enforce single-instance ownership
    if !acquire_single_instance_lock() {
        std::process::exit(0);
    }

    // 2. Launch backend service
    let backend_started = match spawn_backend_service() {
        Ok(_) => {
            // Wait for service readiness
            wait_for_service_readiness(15)
        }
        Err(err) => {
            eprintln!("[JARVIS Desktop] Backend launch error: {}", err);
            false
        }
    };

    if !backend_started {
        eprintln!("[JARVIS Desktop] Service initialization failed. Cleaning up...");
        cleanup_and_shutdown();
        std::process::exit(1);
    }

    // 3. Register POSIX signal handlers
    unsafe {
        libc::signal(libc::SIGINT, sig_handler as usize);
        libc::signal(libc::SIGTERM, sig_handler as usize);
    }

    tauri::Builder::default()
        .setup(|app| {
            if cfg!(debug_assertions) {
                app.handle().plugin(
                    tauri_plugin_log::Builder::default()
                        .level(log::LevelFilter::Info)
                        .build(),
                )?;
            }
            Ok(())
        })
        .on_window_event(|_window, event| {
            if let WindowEvent::CloseRequested { .. } = event {
                println!("[JARVIS Desktop] Window close requested by user.");
                cleanup_and_shutdown();
            }
        })
        .build(tauri::generate_context!())
        .expect("error while building tauri application")
        .run(|_app_handle, event| {
            if let tauri::RunEvent::ExitRequested { .. } = event {
                println!("[JARVIS Desktop] Application exit requested.");
                cleanup_and_shutdown();
            }
        });
}

extern "C" fn sig_handler(_sig: i32) {
    cleanup_and_shutdown();
    std::process::exit(0);
}
