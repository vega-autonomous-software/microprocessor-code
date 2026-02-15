use std::sync::{mpsc, Arc, atomic::{AtomicBool, Ordering}};
use std::thread;
use std::time::Duration;

mod pythonspawn;
mod telemetry_listener;
mod video_listener;

use pythonspawn::runpythonfile_stream;

fn main() {
    println!("Background worker starting...");

    let running = Arc::new(AtomicBool::new(true));
    let r = running.clone();

    ctrlc::set_handler(move || {
        println!("Shutdown signal received.");
        r.store(false, Ordering::SeqCst);
    }).expect("Error setting Ctrl-C handler");

    let (tx, rx) = mpsc::channel::<(&'static str, String)>();

    let run_clone = running.clone();
    thread::spawn(move || {
        telemetry_listener::start_telemetry_listener(run_clone);
    });

    let run_clone = running.clone();
    thread::spawn(move || {
        video_listener::start_video_listener(run_clone);
    });

    let tx_logger = tx.clone();
    thread::spawn(move || {
        runpythonfile_stream(
            "background/python/logger.py",
            "logger.py",
            tx_logger,
        );
    });

    drop(tx);

    while running.load(Ordering::SeqCst) {
        if let Ok((tag, line)) = rx.recv_timeout(Duration::from_millis(500)) {
            println!("[{tag}] {line}");
        }
    }

    println!("Background worker shutting down.");
}
