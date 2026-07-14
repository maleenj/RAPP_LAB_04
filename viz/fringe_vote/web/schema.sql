-- Fringe voting backend — run once in Hostinger phpMyAdmin (SQL tab → paste → Go).
-- Safe to re-run: uses IF NOT EXISTS and an idempotent seed row.

CREATE TABLE IF NOT EXISTS votes (
  client_id  VARCHAR(64) NOT NULL,
  choice     ENUM('mirror','contrast') NOT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (client_id),
  KEY idx_updated (updated_at)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

-- Single-row table holding the roulette's currently selected genre (optional feature).
CREATE TABLE IF NOT EXISTS genre (
  id         TINYINT NOT NULL,
  value      VARCHAR(64) DEFAULT NULL,
  updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP ON UPDATE CURRENT_TIMESTAMP,
  PRIMARY KEY (id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4;

INSERT INTO genre (id, value) VALUES (1, NULL)
  ON DUPLICATE KEY UPDATE id = id;
