use std::fs::OpenOptions;
use std::io::{Read, Write};
use std::net::TcpListener;
use std::sync::{
    Arc,
    atomic::{AtomicBool, Ordering},
};

pub fn start_telemetry_listener(running: Arc<AtomicBool>) {
    let listener =
        TcpListener::bind("0.0.0.0:83").expect("Failed to bind port 83");

    println!("Telemetry listener running on port 83");

    listener
        .set_nonblocking(true)
        .expect("Cannot set non-blocking");

    while running.load(Ordering::SeqCst) {
        if let Ok((mut stream, _)) = listener.accept() {
            let mut file = OpenOptions::new()
                .create(true)
                .append(true)
                .open("telemetry.bin")
                .expect("Cannot open telemetry.bin");

            let mut buffer = [0u8; 12];

            while running.load(Ordering::SeqCst) {
                match stream.read_exact(&mut buffer) {
                    Ok(_) => {
                        file.write_all(&buffer).unwrap();
                    }
                    Err(_) => break,
                }
            }
        }
    }

    println!("Telemetry listener stopped.");
}
