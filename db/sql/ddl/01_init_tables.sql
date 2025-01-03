-- Enable pgvector extension (if not already enabled)
CREATE EXTENSION IF NOT EXISTS vector;

-- Projects table to store different RAG projects
CREATE TABLE IF NOT EXISTS projects (
    id SERIAL PRIMARY KEY,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT uix_project_name UNIQUE (name)
);

-- Files table to store source documents
CREATE TABLE IF NOT EXISTS files (
    id SERIAL PRIMARY KEY,
    project_id INTEGER REFERENCES projects(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(1024) NOT NULL,
    crc VARCHAR(32) NOT NULL,
    file_size BIGINT NOT NULL,
    last_updated TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    last_ingested TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(project_id, file_path)
);

-- Chunks table to store embeddings and text chunks
CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    content TEXT NOT NULL,
    embedding vector(1536),  -- OpenAI's default embedding size
    chunk_index INTEGER NOT NULL,
    chunk_metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(file_id, chunk_index)
);

-- Create indexes for better query performance
CREATE INDEX IF NOT EXISTS idx_files_project_id ON files(project_id);
CREATE INDEX IF NOT EXISTS idx_chunks_file_id ON chunks(file_id);
CREATE INDEX IF NOT EXISTS idx_chunks_embedding ON chunks USING ivfflat (embedding vector_cosine_ops);

-- Create a function to update timestamps
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ language 'plpgsql';

-- Create triggers to automatically update timestamps
CREATE TRIGGER update_projects_updated_at
    BEFORE UPDATE ON projects
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_files_updated_at
    BEFORE UPDATE ON files
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

CREATE TRIGGER update_chunks_updated_at
    BEFORE UPDATE ON chunks
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
