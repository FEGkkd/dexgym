"""Identify ROS schemas by embedded definition and select decoders by version."""
import base64
from dataclasses import dataclass
from hashlib import sha256

from mcap_ros2.decoder import DecoderFactory


class DecodeIssue(ValueError):
    def __init__(self, code, detail):
        super().__init__(detail)
        self.code = code


@dataclass(frozen=True)
class Identity:
    name: str
    schema_encoding: str
    message_encoding: str
    definition_sha256: str

    @property
    def key(self):
        return '|'.join((self.name,self.schema_encoding,self.message_encoding,self.definition_sha256))


def identify(encoding, schema):
    if schema is None:
        return None
    return Identity(schema.name,schema.encoding,encoding,sha256(schema.data).hexdigest())


def declared_support(identity):
    return bool(identity and identity.schema_encoding=='ros2msg' and identity.message_encoding=='cdr')


class EmbeddedDecoders:
    def __init__(self):
        self._functions = {}

    def decode(self, encoding, schema, payload):
        identity = identify(encoding,schema)
        if not declared_support(identity):
            raise DecodeIssue('UNSUPPORTED_SCHEMA',str(identity))
        if len(payload)<4 or payload[:2] not in {b'\x00\x00',b'\x00\x01'}:
            raise DecodeIssue('CDR_DECODE_FAILED','Missing or unsupported CDR encapsulation')
        if identity not in self._functions:
            try:
                # The upstream factory keys by file-local schema.id. Isolate each definition.
                self._functions[identity] = DecoderFactory().decoder_for(encoding,schema)
            except Exception as error:
                raise DecodeIssue('SCHEMA_COMPILE_FAILED',f'{type(error).__name__}: {error}') from error
        try:
            return self._functions[identity](payload)
        except Exception as error:
            raise DecodeIssue('CDR_DECODE_FAILED',f'{type(error).__name__}: {error}') from error


def schema_record(identity, schema):
    return dict(key=identity.key,name=identity.name,message_encoding=identity.message_encoding,
                schema_encoding=identity.schema_encoding,sha256=identity.definition_sha256,
                definition_base64=base64.b64encode(schema.data).decode('ascii'),
                decoder='embedded_ros2msg_cdr_v1' if declared_support(identity) else None)
