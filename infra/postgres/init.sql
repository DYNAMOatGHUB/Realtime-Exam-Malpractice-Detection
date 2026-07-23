-- ============================================================
-- ExamGuard AI — PostgreSQL Initial Schema
-- Run automatically by Docker Compose via the postgres init dir
-- ============================================================

-- Roles seed (Django will create its own tables via migrations,
-- but we pre-create the roles enum and extension here)

CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";   -- for full-text similarity searches

-- Log that schema init ran
DO $$
BEGIN
  RAISE NOTICE 'ExamGuard AI: PostgreSQL init.sql executed at %', NOW();
END $$;
