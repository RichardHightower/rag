"""Database models for RAG."""

from datetime import datetime, timezone as tz
from sqlalchemy import Column, Integer, String, Text, DateTime, ForeignKey, BigInteger, UniqueConstraint
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import declarative_base, relationship
from pgvector.sqlalchemy import Vector

# Use the new SQLAlchemy 2.0 style declarative base
Base = declarative_base()


def utc_now():
    """Get current UTC datetime."""
    return datetime.now(tz.utc)


class Project(Base):
    """Project model."""

    __tablename__ = 'projects'
    __table_args__ = (
        UniqueConstraint('name', name='uix_project_name'),
    )

    id = Column(Integer, primary_key=True)
    name = Column(String(255), nullable=False, unique=True)
    description = Column(Text)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now)

    files = relationship('File', back_populates='project', cascade='all, delete-orphan')


class File(Base):
    """File model."""

    __tablename__ = 'files'
    __table_args__ = (
        UniqueConstraint('filename', 'file_path', 'project_id', name='uix_file_path_project'),
    )

    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id', ondelete='CASCADE'))
    filename = Column(String, nullable=False)
    file_path = Column(String, nullable=False)
    file_type = Column(String)
    crc = Column(BigInteger, nullable=False)
    file_size = Column(BigInteger, nullable=False)
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    project = relationship('Project', back_populates='files')
    chunks = relationship('Chunk', back_populates='file', cascade='all, delete-orphan')


class Chunk(Base):
    """Chunk model with vector embedding."""

    __tablename__ = 'chunks'

    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id', ondelete='CASCADE'))
    content = Column(Text, nullable=False)
    start_char = Column(Integer, nullable=False)
    end_char = Column(Integer, nullable=False)
    embedding = Column(Vector(1536))  # Default to OpenAI's dimension
    chunk_index = Column(Integer, nullable=False)
    chunk_metadata = Column(JSONB, default={})
    created_at = Column(DateTime(timezone=True), default=utc_now)
    updated_at = Column(DateTime(timezone=True), default=utc_now, onupdate=utc_now)

    file = relationship('File', back_populates='chunks')
