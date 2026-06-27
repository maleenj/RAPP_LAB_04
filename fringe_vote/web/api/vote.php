<?php
// POST {clientId, choice}  — record/replace this device's current vote.
require __DIR__ . '/db.php';
cors();

$in = body_json();
$clientId = $in['clientId'] ?? ($_POST['clientId'] ?? '');
$choice   = $in['choice']   ?? ($_POST['choice']   ?? '');

$clientId = substr(trim((string)$clientId), 0, 64);
if ($clientId === '' || !in_array($choice, ['mirror', 'contrast'], true)) {
    json_out(['error' => 'bad request — need clientId and choice in {mirror,contrast}'], 400);
}

$sql = "INSERT INTO votes (client_id, choice) VALUES (:id, :c)
        ON DUPLICATE KEY UPDATE choice = VALUES(choice), updated_at = CURRENT_TIMESTAMP";
db()->prepare($sql)->execute([':id' => $clientId, ':c' => $choice]);

json_out(['ok' => true]);
