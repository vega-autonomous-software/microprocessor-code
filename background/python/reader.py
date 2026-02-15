from schema import TelemetryPacket, STRUCT_SIZE


class BinaryReader:
    def __init__(self, path):
        self.file = open(path, "rb")

    def read_next(self):
        data = self.file.read(STRUCT_SIZE)
        if not data:
            return None
        return TelemetryPacket.from_bytes(data)

    def close(self):
        self.file.close()
