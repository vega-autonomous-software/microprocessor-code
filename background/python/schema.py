import struct
from dataclasses import dataclass

BINARY_FORMAT = "<Qfff"
STRUCT_SIZE = struct.calcsize(BINARY_FORMAT)


@dataclass
class TelemetryPacket:
    timestamp_us: int
    accelerator: float
    brake: float
    steering: float

    def to_bytes(self) -> bytes:
        return struct.pack(
            BINARY_FORMAT,
            self.timestamp_us,
            self.accelerator,
            self.brake,
            self.steering,
        )

    @staticmethod
    def from_bytes(data: bytes):
        values = struct.unpack(BINARY_FORMAT, data)
        return TelemetryPacket(*values)
