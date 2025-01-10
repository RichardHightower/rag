-- Drop existing indexes if they exist
DROP INDEX IF EXISTS idx_chunks_embedding;

-- Create optimized IVFFlat index for vector similarity search
-- IVFFlat is particularly efficient for cosine similarity searches in high dimensions
-- The 'lists' parameter determines the number of clusters for the index
-- A good rule of thumb is sqrt(N)/2 where N is the expected number of vectors
-- We use 100 as a reasonable default that can be tuned based on data size
CREATE INDEX idx_chunks_embedding ON chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);

-- Create supporting index for efficient file and chunk lookups
CREATE INDEX IF NOT EXISTS idx_chunks_file_metadata ON chunks
USING btree (file_id, chunk_index);

-- Update table statistics for query optimization
ANALYZE chunks;

-- Note: The IVFFlat index type is chosen because:
-- 1. It's optimized for ANN (Approximate Nearest Neighbor) searches
-- 2. Works well with cosine similarity (<=> operator)
-- 3. Provides good balance between search speed and accuracy
-- 4. Performs well with high-dimensional vectors (e.g., OpenAI embeddings)