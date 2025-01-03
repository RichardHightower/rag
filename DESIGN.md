Below is a **Business Requirements document**, followed by a **Product Requirements & Design document** (PRD/Design Doc) and then a **working code example**. This forms the basic design goals for the RAG system. This code demonstrates ingesting files, chunking text, generating embeddings (via OpenAI, Hugging Face, or a mock interface), and storing those embeddings in a Postgres database using pgvector.

---

## 1. Business Requirements

- **Goal**: Build a system that reads text files (source code, Markdown, plain text, etc.), chunks them (with overlap), generates vector embeddings, and stores the chunks + embeddings in a Postgres database that has the pgvector extension installed.
- **Database Schema**:
    - **Projects** to group files.
    - **Files** to store metadata (filename, CRC, timestamps, file size).
    - **Chunks** to store text chunks and their associated embeddings (pgvector column).
- **Embeddings**:
    1. **OpenAI** API (with configurable model/dimension, e.g., `text-embedding-3-small` or `text-embedding-3-large`).
    2. **HuggingFace / Sentence Transformers**.
    3. **Mock** (for testing without hitting an external API).
- **Dimension Control**:
    - Ability to specify dimension in a config and ensure the DB’s pgvector column is updated (via `ALTER TABLE` if needed).
    - Potential to zero-pad or truncate if the dimension changes.
- **Chunking**:
    - Overlapping line-based chunking (e.g., 500 lines with 50-line overlap, but this should be configurable).
- **Implementation**:
    - A Python codebase that sets up an embedding interface (`Embedder`) with concrete classes for OpenAI, Hugging Face, and a mock embedder.
        - Dimension size and model should be configurable
    - A main handler (`DBFileHandler`) that ingests files, chunks them, calls the embedder, and writes to the DB.

---

## 2. Product Requirements & Design Document (PRD/Design Doc)

### 2.1 Overview & Objectives

**Objective**:

Create a retrieval pipeline that stores text embeddings in Postgres using pgvector, enabling semantic search or question-answering over large text files.

**Key Features**:

1. **Configurable Embedding Source**:
    - **OpenAI** (API-based) with dynamic dimension & model.
    - **HuggingFace** (local inference) with dynamic dimension & model.
    - **Mock** for testing or dev environments.
2. **Data Chunking**:
    - Split text files into overlapping chunks for better context retrieval.
3. **Database**:
    - **pgvector** extension for storing embeddings.
    - Tables for `projects`, `files`, and `chunks`.
    - Automatic dimension checks/updates at startup.
4. **File Lifecycle**:
    - `add_file(project_id, file_path)`: Read, chunk, embed, store.
    - `remove_file(project_id, file_id)`: Remove a file and its chunks.

### 2.2 Scope & Constraints

- We assume files are **plain text** (converted from PDF if needed).
- Overlap chunking is line-based (extendable to tokens if desired).
- We rely on external libraries for embeddings (OpenAI or Hugging Face).
- Changes to embedding dimensions may require data migration, re-embedding, or zero-padding/truncation.

### 2.3 Technical Stack

1. **Python 3.12+**
2. **SQLAlchemy** + **sqlalchemy-pgvector** for ORM and vector columns.
3. **psycopg2-binary** for Postgres connections.
4. **OpenAI** library for API-based embeddings.
5. **sentence-transformers** for Hugging Face embeddings.
6. **pgvector** extension in Postgres (`CREATE EXTENSION vector;`).

### 2.4 Data Model

**projects**

- `id` (Primary Key)
- `name` (String)

**files**

- `id` (Primary Key)
- `project_id` → Foreign Key → `projects.id`
- `filename` (String)
- `crc` (String)
- `last_updated` (DateTime)
- `last_ingested` (DateTime)
- `file_size` (Float)

**chunks**

- `id` (Primary Key)
- `file_id` → Foreign Key → `files.id` (on delete cascade)
- `vector` → `vector(<dimension>)` column (alterable via SQL DDL on startup)
- `data` (Text) → The chunked text content

### 2.5 Workflow Overview

1. **Startup**:
    1. Load config (model name, dimension, DB URL, etc.).
    2. Initialize the chosen embedder (OpenAI, HuggingFace, or Mock).
    3. **Check/alter** the `chunks.vector` column dimension (via SQL DDL).
2. **Ingest File** (`add_file`):
    1. Read file → chunk the content.
    2. Generate embeddings for each chunk.
    3. Insert metadata into `files`, chunk records into `chunks`.
3. **Remove File** (`remove_file`):
    1. Find the file by `file_id` & `project_id`.
    2. Delete file and cascade-delete its chunks.
4. **Future**:
    - Use the stored embeddings for similarity search or Q&A retrieval.

### 2.6 Edge Cases & Future Considerations

- **Dimension Mismatch**: If the embedder’s dimension doesn’t match the DB column, we run `ALTER TABLE`. For existing data, consider re-embedding or padding/truncating old vectors.
- **Performance**: For large files, chunking + embedding can be CPU- and API-intensive. Batch requests or asynchronous ingestion might be required in production.
- **Security**: Ensure the DB and OpenAI credentials are properly secured.

---

## 3. Working Code Example

Below is **one** possible Python script illustrating:

1. **Embedder Interface**.
2. **Concrete** `OpenAIEmbedder`, `HuggingFaceEmbedder`, and `MockEmbedder`.
3. A **DBFileHandler** class that orchestrates file ingestion/deletion.
4. **Startup** dimension checks on the `chunks.vector` column.

> Important:
> 
> - Install required packages:
>     
>     ```bash
>     pip install openai sentence-transformers sqlalchemy psycopg2-binary sqlalchemy-pgvector
>     
>     ```
>     
> - Ensure `CREATE EXTENSION vector;` in your Postgres DB.

```python
import os
import hashlib
import datetime
import random
from typing import List

# 1) External Libraries
import openai
from sentence_transformers import SentenceTransformer

from sqlalchemy import (
    create_engine,
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text,
    text
)
from sqlalchemy.orm import relationship, sessionmaker
from sqlalchemy.ext.declarative import declarative_base

from sqlalchemy_pgvector import Vector
from abc import ABC, abstractmethod

# -------------------------------------------------------------------
# 2) EMBEDDING INTERFACE + CONCRETE IMPLEMENTATIONS
# -------------------------------------------------------------------
class Embedder(ABC):
    """
    Base class for all embedders. Must specify model_name, dimension,
    and embed_texts() method.
    """
    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

class OpenAIEmbedder(Embedder):
    """
    Embeds text using OpenAI's API in batches.
    
    Args:
        model_name (str): e.g., "text-embedding-3-small", "text-embedding-3-large"
        dimension (int): e.g., 512, 1536, 3072, etc. (depends on the model)
        api_key (str): Your OpenAI API key
        batch_size (int): Number of text chunks to embed per request
    """
    def __init__(
        self, 
        model_name: str, 
        dimension: int, 
        api_key: str, 
        batch_size: int = 16
    ):
        super().__init__(model_name, dimension)
        openai.api_key = api_key
        self.batch_size = batch_size

    def get_dimension(self) -> int:
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts in batches to minimize API calls.
        Note: Ensure that each text chunk is within token limits 
              (e.g., ~8191 tokens for 'text-embedding-ada-002') if very large.
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            # The OpenAI Embedding endpoint supports a list of inputs in 'input'
            response = openai.Embedding.create(
                input=batch,
                model=self.model_name
            )
            # Response has one embedding per item in 'data', in the same order
            for item in response["data"]:
                vec = item["embedding"]
                # Optionally pad or truncate if you expect dimension mismatches
                all_embeddings.append(vec)

        return all_embeddings

class HuggingFaceEmbedder(Embedder):
    """
    Embeds text using a Sentence Transformer model from Hugging Face.
    model_name: e.g. "sentence-transformers/all-MiniLM-L6-v2"
    dimension: must match or you must handle padding/truncation if there's a mismatch
    """
    def __init__(self, model_name: str, dimension: int):
        super().__init__(model_name, dimension)
        self.model = SentenceTransformer(model_name)

    def get_dimension(self) -> int:
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        arr = self.model.encode(texts, convert_to_numpy=True)
        return arr.tolist()

class MockEmbedder(Embedder):
    """
    A mock embedder that returns random float vectors.
    Useful for testing without external dependencies.
    """
    def get_dimension(self) -> int:
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        results = []
        for _ in texts:
            vec = [random.random() for _ in range(self.dimension)]
            results.append(vec)
        return results

# -------------------------------------------------------------------
# 3) DATABASE MODELS
# -------------------------------------------------------------------
Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    files = relationship("File", back_populates="project")

class File(Base):
    __tablename__ = 'files'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    filename = Column(String, nullable=False)
    crc = Column(String, nullable=False)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)
    last_ingested = Column(DateTime, default=datetime.datetime.utcnow)
    file_size = Column(Float)

    project = relationship("Project", back_populates="files")
    chunks = relationship("Chunk", back_populates="file", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)

    # We'll initialize with a default dimension, but alter at runtime if needed.
    vector = Column(Vector(512))
    data = Column(Text, nullable=False)

    file = relationship("File", back_populates="chunks")

# -------------------------------------------------------------------
# 4) UTILITY FOR ALTERING VECTOR DIMENSIONS
# -------------------------------------------------------------------
def ensure_vector_dimension(engine, desired_dim: int):
    """
    If the 'chunks.vector' column dimension is different, attempt an ALTER TABLE.
    You may need to re-embed or pad/truncate existing data for production usage.
    """
    with engine.connect() as conn:
        # Ensure the extension is enabled
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        # Attempt to alter the column dimension
        alter_sql = f"""
            ALTER TABLE chunks
            ALTER COLUMN vector TYPE vector({desired_dim});
        """
        try:
            conn.execute(text(alter_sql))
            print(f"[INFO] Successfully set vector dimension to {desired_dim}")
        except Exception as e:
            print(f"[WARNING] Could not alter dimension to {desired_dim}: {e}")

# -------------------------------------------------------------------
# 5) CHUNKING FUNCTION
# -------------------------------------------------------------------
def chunk_text(content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Overlapping line-based chunking.
    e.g. chunk_size=500, overlap=50 -> each chunk has 500 lines,
    next chunk starts 450 lines after the previous chunk start.
    """
    lines = content.splitlines()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(lines), step):
        chunk = lines[i:i+chunk_size]
        chunks.append('\n'.join(chunk))
    return chunks

# -------------------------------------------------------------------
# 6) FILE HANDLER FOR DB OPERATIONS
# -------------------------------------------------------------------
class DBFileHandler:
    def __init__(self, db_url: str, embedder: Embedder):
        self.embedder = embedder
        self.engine = create_engine(db_url)

        # 1) Ensure dimension is correct on 'chunks.vector'
        desired_dim = self.embedder.get_dimension()
        ensure_vector_dimension(self.engine, desired_dim)

        # 2) Create tables if not exist
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)
        
    def create_project(self, project_name: str): 
	    # TODO implement this
	    pass
	  
	  def delete_project(self, project_name: str): 
	    # Delete all files and chunks associated with this project
	    pass  

    def add_file(self, project_id: int, file_path: str):
        session = self.Session()
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # Generate metadata
            file_crc = hashlib.md5(content.encode('utf-8')).hexdigest()
            file_size = len(content)

            # Check project
            project = session.query(Project).filter_by(id=project_id).first()
            if not project:
                raise ValueError(f"No project found with ID {project_id}")

            # Insert file
            file_entry = File(
                project_id=project_id,
                filename=file_path,
                crc=file_crc,
                file_size=file_size,
                last_updated=datetime.datetime.utcnow(),
                last_ingested=datetime.datetime.utcnow()
            )
            session.add(file_entry)
            session.flush()  # get file_entry.id

            # Chunk TODO read chunk_size and overlap from file.
            text_chunks = chunk_text(content, chunk_size=500, overlap=50)
            # Embed
            embeddings = self.embedder.embed_texts(text_chunks)
            if len(embeddings) != len(text_chunks):
                raise ValueError("[ERROR] Mismatch between chunk count & embedding count")

            # Store
            for chunk_str, vector in zip(text_chunks, embeddings):
                chunk_entry = Chunk(file_id=file_entry.id, vector=vector, data=chunk_str)
                session.add(chunk_entry)

            session.commit()
            print(f"[INFO] Ingested file '{file_path}' into project {project_id}.")
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def remove_file(self, project_id: int, file_id: int):
        session = self.Session()
        try:
            file_entry = (
                session.query(File)
                .filter_by(id=file_id, project_id=project_id)
                .first()
            )
            if not file_entry:
                raise ValueError(f"File ID={file_id} not found in project={project_id}")
            session.delete(file_entry)
            session.commit()
            print(f"[INFO] Removed file ID={file_id} from project ID={project_id}")
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

# -------------------------------------------------------------------
# 7) EXAMPLE USAGE
# -------------------------------------------------------------------
if __name__ == "__main__":
    # Example config
    config = {
        "database_url": "postgresql+psycopg2://user:password@localhost:5432/mydb",
        "openai_api_key": "YOUR_OPENAI_API_KEY",
        "model_name": "text-embedding-3-small",  # or "text-embedding-3-large"
        "dimension": 512
    }

    # Select which embedder to use:
    # 1) OpenAI
    embedder = OpenAIEmbedder(
        model_name=config["model_name"],
        dimension=config["dimension"],
        api_key=config["openai_api_key"]
    )

    # 2) Hugging Face Example (comment above, uncomment below to switch):
    # embedder = HuggingFaceEmbedder(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     dimension=384
    # )

    # 3) Mock Example (comment above, uncomment below to switch):
    # embedder = MockEmbedder(
    #     model_name="mock",
    #     dimension=128
    # )

    # Instantiate DBFileHandler
    db_url = config["database_url"]
    handler = DBFileHandler(db_url=db_url, embedder=embedder)

    # Suppose there's a project with ID=1
    project_id = 1
    file_path = "example.txt"

    # Add the file
    handler.add_file(project_id, file_path)

    # Remove (once you know the file_id):
    # handler.remove_file(project_id, file_id=some_file_id)

```

---

## 4. Final Notes

- **Dimension Changes**: If switching from 512 → 1536 or vice versa, you’ll want to re-embed or pad/truncate existing data. The code shows how to run `ALTER TABLE` but does not handle data migration for existing rows. This is out of scope for now. If you change the embedding, you will need to re-import the files and restart the project.
- **Security**: Manage your secrets (OpenAI API key, DB creds) securely, e.g., environment variables or a vault.

This code provides a **starting point** for a pluggable text-ingestion pipeline with chunking, embedding, and vector storage in PostgreSQL (pgvector). Adjust or extend as needed for your production requirements.

---

Below is a **suggested project layout** that breaks the code into multiple modules/files. This helps keep your code organized, maintainable, and testable.

---

# 1. Recommended Directory Structure

```
my_project/
├── README.md
├── requirements.txt
├── setup.py                 # (Optional) if you plan to package/install
└── src/
    ├── my_app/
    │   ├── __init__.py
    │   ├── main.py
    │   ├── config.py
    │   ├── embeddings/
    │   │   ├── __init__.py
    │   │   ├── base.py
    │   │   ├── openai_embedder.py
    │   │   ├── huggingface_embedder.py
    │   │   └── mock_embedder.py
    │   ├── db/
    │   │   ├── __init__.py
    │   │   ├── models.py
    │   │   ├── dimension_utils.py
    │   │   ├── chunking.py
    │   │   └── db_file_handler.py
    │   └── scripts/
    │       └── run_example.py
    └── tests/
        ├── __init__.py
        ├── test_embeddings.py
        ├── test_db_handler.py
        └── ...

```

Here’s a quick overview:

- **`my_app/embeddings/`**:
    - **`base.py`**: Contains the abstract `Embedder` class definition.
    - **`openai_embedder.py`**: Concrete class for OpenAI embedding (batching included).
    - **`huggingface_embedder.py`**: Concrete class for Hugging Face embedding.
    - **`mock_embedder.py`**: Concrete class for a mock embedder.
- **`my_app/db/`**:
    - **`models.py`**: SQLAlchemy models (`Project`, `File`, `Chunk`) and `Base`.
    - **`dimension_utils.py`**: The `ensure_vector_dimension(...)` function (and any dimension-altering logic).
    - **`chunking.py`**: The `chunk_text(...)` function.
    - **`db_file_handler.py`**: The `DBFileHandler` class, containing methods for creating/removing projects, ingesting files, removing files, etc.
- **`my_app/main.py`**:
    - Could parse config or environment variables, build the embedder, and run the handler.
- **`my_app/scripts/run_example.py`**:
    - A script showing how to instantiate everything and do an example run (similar to the `__main__` block we previously had).
- **`tests/`**:
    - Directory for test files (e.g. `test_embeddings.py`, `test_db_handler.py`).

---

# 2. Example Code in Modules

Below is a **sample** of how you might split up the code into separate files. We’ll include the essential pieces from your existing code, with imports adjusted to reflect this new structure.

> Note: This is a template. Adjust the imports / relative paths (from ... import ...) as needed depending on your exact folder names.
> 

---

### `my_app/embeddings/base.py`

```python
# src/my_app/embeddings/base.py

from abc import ABC, abstractmethod
from typing import List

class Embedder(ABC):
    """
    Base class for all embedders. Must specify model_name, dimension,
    and embed_texts() method.
    """
    def __init__(self, model_name: str, dimension: int):
        self.model_name = model_name
        self.dimension = dimension

    @abstractmethod
    def get_dimension(self) -> int:
        pass

    @abstractmethod
    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        pass

```

---

### `my_app/embeddings/openai_embedder.py`

```python
# src/my_app/embeddings/openai_embedder.py

import openai
from typing import List

from .base import Embedder

class OpenAIEmbedder(Embedder):
    """
    Embeds text using OpenAI's API in batches.

    Args:
        model_name (str): e.g., "text-embedding-3-small", "text-embedding-3-large"
        dimension (int): e.g., 512, 1536, 3072, etc. (depends on the model)
        api_key (str): Your OpenAI API key
        batch_size (int): Number of text chunks to embed per request
    """
    def __init__(
        self,
        model_name: str,
        dimension: int,
        api_key: str,
        batch_size: int = 16
    ):
        super().__init__(model_name, dimension)
        openai.api_key = api_key
        self.batch_size = batch_size

    def get_dimension(self) -> int:
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embeds a list of texts in batches to minimize API calls.
        Ensure each text chunk is within OpenAI's token limit if they are large.
        """
        all_embeddings = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i:i + self.batch_size]
            response = openai.Embedding.create(
                input=batch,
                model=self.model_name
            )
            # 'data' has an embedding for each item in 'batch'
            for item in response["data"]:
                vec = item["embedding"]
                all_embeddings.append(vec)

        return all_embeddings

```

---

### `my_app/embeddings/huggingface_embedder.py`

```python
# src/my_app/embeddings/huggingface_embedder.py

from typing import List
from sentence_transformers import SentenceTransformer

from .base import Embedder

class HuggingFaceEmbedder(Embedder):
    """
    Embeds text using a Sentence Transformer model from Hugging Face.
    """
    def __init__(self, model_name: str, dimension: int):
        super().__init__(model_name, dimension)
        self.model = SentenceTransformer(model_name)

    def get_dimension(self) -> int:
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        arr = self.model.encode(texts, convert_to_numpy=True)
        # If self.dimension differs from actual model dimension,
        # consider padding/truncation.
        return arr.tolist()

```

---

### `my_app/embeddings/mock_embedder.py`

```python
# src/my_app/embeddings/mock_embedder.py

import random
from typing import List
from .base import Embedder

class MockEmbedder(Embedder):
    """
    A mock embedder that returns random float vectors.
    Useful for testing without external dependencies.
    """
    def get_dimension(self) -> int:
        return self.dimension

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        results = []
        for _ in texts:
            vec = [random.random() for _ in range(self.dimension)]
            results.append(vec)
        return results

```

---

### `my_app/db/models.py`

```python
# src/my_app/db/models.py

import datetime
from sqlalchemy import (
    Column,
    Integer,
    String,
    Float,
    DateTime,
    ForeignKey,
    Text
)
from sqlalchemy.orm import relationship, declarative_base
from sqlalchemy_pgvector import Vector

Base = declarative_base()

class Project(Base):
    __tablename__ = 'projects'
    id = Column(Integer, primary_key=True)
    name = Column(String, nullable=False)

    files = relationship("File", back_populates="project")

class File(Base):
    __tablename__ = 'files'
    id = Column(Integer, primary_key=True)
    project_id = Column(Integer, ForeignKey('projects.id'))
    filename = Column(String, nullable=False)
    crc = Column(String, nullable=False)
    last_updated = Column(DateTime, default=datetime.datetime.utcnow)
    last_ingested = Column(DateTime, default=datetime.datetime.utcnow)
    file_size = Column(Float)

    project = relationship("Project", back_populates="files")
    chunks = relationship("Chunk", back_populates="file", cascade="all, delete-orphan")

class Chunk(Base):
    __tablename__ = 'chunks'
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey('files.id'), nullable=False)

    # Default dimension of 512 (can be altered at runtime)
    vector = Column(Vector(512))
    data = Column(Text, nullable=False)

    file = relationship("File", back_populates="chunks")

```

---

### `my_app/db/dimension_utils.py`

```python
# src/my_app/db/dimension_utils.py

from sqlalchemy import text

def ensure_vector_dimension(engine, desired_dim: int):
    """
    If the 'chunks.vector' column dimension is different, attempt an ALTER TABLE.
    In real usage, handle data migration or re-embedding carefully if dimension changes.
    """
    with engine.connect() as conn:
        # Enable extension if missing
        conn.execute(text("CREATE EXTENSION IF NOT EXISTS vector;"))

        alter_sql = f"""
            ALTER TABLE chunks
            ALTER COLUMN vector TYPE vector({desired_dim});
        """
        try:
            conn.execute(text(alter_sql))
            print(f"[INFO] Successfully set vector dimension to {desired_dim}")
        except Exception as e:
            print(f"[WARNING] Could not alter dimension to {desired_dim}: {e}")

```

---

### `my_app/db/chunking.py`

```python
# src/my_app/db/chunking.py

from typing import List

def chunk_text(content: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Overlapping line-based chunking.
    chunk_size=500, overlap=50 => each chunk has 500 lines,
    next chunk starts 450 lines after the previous chunk start.
    """
    lines = content.splitlines()
    chunks = []
    step = max(chunk_size - overlap, 1)
    for i in range(0, len(lines), step):
        chunk = lines[i:i+chunk_size]
        chunks.append('\n'.join(chunk))
    return chunks

```

---

### `my_app/db/db_file_handler.py`

```python
# src/my_app/db/db_file_handler.py

import hashlib
import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from .models import Base, Project, File, Chunk
from .dimension_utils import ensure_vector_dimension
from .chunking import chunk_text

class DBFileHandler:
    def __init__(self, db_url: str, embedder):
        self.embedder = embedder
        self.engine = create_engine(db_url)

        # Ensure dimension matches embedder
        desired_dim = self.embedder.get_dimension()
        ensure_vector_dimension(self.engine, desired_dim)

        # Create tables if they don't exist
        Base.metadata.create_all(self.engine)
        self.Session = sessionmaker(bind=self.engine)

    def create_project(self, project_name: str):
        """
        Create a new project (if not exists).
        """
        session = self.Session()
        try:
            # Check if project already exists
            existing = session.query(Project).filter_by(name=project_name).first()
            if existing:
                print(f"[INFO] Project '{project_name}' already exists (ID: {existing.id}).")
                return existing.id

            project = Project(name=project_name)
            session.add(project)
            session.commit()
            print(f"[INFO] Created project '{project_name}' with ID={project.id}.")
            return project.id
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def delete_project(self, project_name: str):
        """
        Delete a project by name, including all files and chunks (cascade).
        """
        session = self.Session()
        try:
            proj = session.query(Project).filter_by(name=project_name).first()
            if not proj:
                print(f"[WARNING] No project found with name '{project_name}'.")
                return
            session.delete(proj)
            session.commit()
            print(f"[INFO] Deleted project '{project_name}' (ID={proj.id}) and all related files/chunks.")
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def add_file(self, project_id: int, file_path: str, chunk_size=500, overlap=50):
        """
        Ingest file into the specified project, chunk the text, embed, and store.
        """
        session = self.Session()
        try:
            # Read file
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # File metadata
            file_crc = hashlib.md5(content.encode('utf-8')).hexdigest()
            file_size = len(content)

            # Check project
            project = session.query(Project).filter_by(id=project_id).first()
            if not project:
                raise ValueError(f"No project found with ID={project_id}")

            file_entry = File(
                project_id=project_id,
                filename=file_path,
                crc=file_crc,
                file_size=file_size,
                last_updated=datetime.datetime.utcnow(),
                last_ingested=datetime.datetime.utcnow()
            )
            session.add(file_entry)
            session.flush()  # get file_id

            # Chunk
            text_chunks = chunk_text(content, chunk_size=chunk_size, overlap=overlap)

            # Embed
            embeddings = self.embedder.embed_texts(text_chunks)
            if len(embeddings) != len(text_chunks):
                raise ValueError("[ERROR] Mismatch between chunk count & embedding count")

            # Store
            for chunk_str, vector in zip(text_chunks, embeddings):
                chunk_entry = Chunk(
                    file_id=file_entry.id,
                    vector=vector,
                    data=chunk_str
                )
                session.add(chunk_entry)

            session.commit()
            print(f"[INFO] Ingested '{file_path}' into project ID={project_id} with {len(text_chunks)} chunks.")
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

    def remove_file(self, project_id: int, file_id: int):
        """
        Delete a single file within a project, cascade-deleting its chunks.
        """
        session = self.Session()
        try:
            file_entry = (
                session.query(File)
                .filter_by(id=file_id, project_id=project_id)
                .first()
            )
            if not file_entry:
                raise ValueError(f"File ID={file_id} not found in project={project_id}")

            session.delete(file_entry)
            session.commit()
            print(f"[INFO] Removed file ID={file_id} from project ID={project_id}.")
        except Exception as e:
            session.rollback()
            raise e
        finally:
            session.close()

```

---

### `my_app/scripts/run_example.py`

```python
# src/my_app/scripts/run_example.py

import os
from my_app.db.db_file_handler import DBFileHandler
from my_app.embeddings.openai_embedder import OpenAIEmbedder
from my_app.embeddings.huggingface_embedder import HuggingFaceEmbedder
from my_app.embeddings.mock_embedder import MockEmbedder

def main():
    config = {
        "database_url": "postgresql+psycopg2://user:password@localhost:5432/mydb",
        "openai_api_key": "YOUR_OPENAI_API_KEY",
        "model_name": "text-embedding-3-small",  # or "text-embedding-3-large"
        "dimension": 512
    }

    # 1) Choose embedder
    # example: OpenAI
    embedder = OpenAIEmbedder(
        model_name=config["model_name"],
        dimension=config["dimension"],
        api_key=config["openai_api_key"],
        batch_size=16
    )

    # or Hugging Face example
    # embedder = HuggingFaceEmbedder("sentence-transformers/all-MiniLM-L6-v2", 384)

    # or Mock example
    # embedder = MockEmbedder("mock", 128)

    # 2) Initialize handler
    handler = DBFileHandler(db_url=config["database_url"], embedder=embedder)

    # Create or get a project
    project_name = "DemoProject"
    project_id = handler.create_project(project_name)

    # Add file
    file_path = "example.txt"  # Must exist
    handler.add_file(project_id=project_id, file_path=file_path, chunk_size=500, overlap=50)

    # Optional: remove file (assuming you know the file ID)
    # handler.remove_file(project_id=project_id, file_id=some_file_id)

    # Optional: delete project
    # handler.delete_project(project_name)

if __name__ == "__main__":
    main()

```

---

# 3. Usage Instructions

1. **Install Dependencies** (in a virtual environment, ideally):
    
    ```bash
    pip install -r requirements.txt
    # or
    pip install openai sentence-transformers sqlalchemy psycopg2-binary sqlalchemy-pgvector
    
    ```
    
2. **Ensure** `pgvector` extension is available in your Postgres DB:
    
    ```sql
    CREATE EXTENSION IF NOT EXISTS vector;
    
    ```
    
3. **Run** the example script:

(Adjust paths as needed.)
    
    ```bash
    cd my_project
    python -m src.my_app.scripts.run_example
    
    ```
    

---

## Final Notes

- This structure keeps each concern in its own file:
    - **Embeddings** in `my_app/embeddings/…`
    - **Database** logic (models, dimension checks, chunking, and the handler) in `my_app/db/…`
    - An example runner in `my_app/scripts/run_example.py`.
- You can further separate out configuration logic (e.g. read from a `config.yaml` or environment variables) into `my_app/config.py`.
- For **testing**, create a `tests/` folder parallel to `my_app/` where you can write unit/integration tests targeting each module.

This modular layout should make it easier to **maintain, extend, and test** the system as it grows.