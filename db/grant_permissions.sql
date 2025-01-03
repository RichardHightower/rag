-- Grant necessary permissions to postgres user
ALTER USER postgres WITH SUPERUSER;

-- Grant all privileges on test database
\c vectordb_test;
GRANT ALL PRIVILEGES ON DATABASE vectordb_test TO postgres;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO postgres;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO postgres;
ALTER USER postgres WITH SUPERUSER;
