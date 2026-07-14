-- OPTIONAL research logging — append-only history of every vote, for later analysis.
-- Additive & safe: does NOT change votes/genre tables, vote.php, tally.php, or anything else.
-- Run once in Hostinger phpMyAdmin (SQL tab → paste → Go), BEFORE the show.
--
-- The two triggers copy every vote (new votes, changes, AND the ~10s heartbeats) into
-- vote_events with a millisecond timestamp. `changed = 1` marks a real switch of choice
-- (vs a heartbeat re-send of the same choice), so you can filter either way.
-- client_id is an anonymous random UUID from the phone's browser — no personal data.

CREATE TABLE IF NOT EXISTS vote_events (
  id         BIGINT AUTO_INCREMENT PRIMARY KEY,
  client_id  VARCHAR(64) NOT NULL,
  choice     ENUM('mirror','contrast') NOT NULL,
  changed    TINYINT(1) NOT NULL DEFAULT 0,          -- 1 = choice differed from previous
  created_at TIMESTAMP(3) NOT NULL DEFAULT CURRENT_TIMESTAMP(3),
  KEY idx_created (created_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- New voter's first vote (INSERT) — count as a change (from nothing).
CREATE TRIGGER votes_log_ai AFTER INSERT ON votes FOR EACH ROW
  INSERT INTO vote_events (client_id, choice, changed)
  VALUES (NEW.client_id, NEW.choice, 1);

-- Existing voter re-votes/heartbeats (UPDATE) — flag whether the choice actually changed.
CREATE TRIGGER votes_log_au AFTER UPDATE ON votes FOR EACH ROW
  INSERT INTO vote_events (client_id, choice, changed)
  VALUES (NEW.client_id, NEW.choice, NEW.choice <> OLD.choice);

-- ─────────────────────────────────────────────────────────────────────────────
-- To REMOVE all of this later (full revert, leaves live voting untouched):
--   DROP TRIGGER IF EXISTS votes_log_ai;
--   DROP TRIGGER IF EXISTS votes_log_au;
--   DROP TABLE   IF EXISTS vote_events;
