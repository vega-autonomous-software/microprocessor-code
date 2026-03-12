use std::sync::mpsc;
use std::thread;
use std::time::Duration;

mod pythonspawn;
use pythonspawn::runpythonfile_stream;

fn main() {
    let (tx, rx) = mpsc::channel::<(&'static str, String)>();

    let tx_engine = tx.clone();
    let handle_engine = thread::spawn(move || {
        runpythonfile_stream("python/engine.py", "engine.py", tx_engine);
    });

    thread::sleep(Duration::from_secs(10));

    let tx_manual = tx.clone();
    let handle_manual = thread::spawn(move || {
        runpythonfile_stream(
            "python/manual_drive_sensors.py",
            "manual_drive_sensors.py",
            tx_manual,
        );
    });

    drop(tx);

    while let Ok((tag, line)) = rx.recv() {
        println!("[{tag}] {line}");
    }

    handle_engine.join().expect("engine.py thread panicked");
    handle_manual.join().expect("manual_drive_sensors.py thread panicked");
}