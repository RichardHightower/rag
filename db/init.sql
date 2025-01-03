-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create test database if it doesn't exist
CREATE DATABASE vectordb_test;

-- Switch to test database and set up extensions
\c vectordb_test;
CREATE EXTENSION IF NOT EXISTS vector;

-- Grant necessary permissions
ALTER USER postgres WITH SUPERUSER;
GRANT ALL PRIVILEGES ON DATABASE vectordb_test TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
