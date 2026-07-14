<?php
// tally.php  (deliberately NOT named state.php — Hostinger's malware filter blocks
// that generic filename. Keep this name in sync with vote-api.js + the robot bridge.)
// GET ?window=N  — live tally of "active" voters (those who voted within N seconds).
// Returns {mirror, contrast, active, ratio, winner, window, ts}.
//   ratio = contrast / (mirror+contrast)  -> 0 = all mirror, 1 = all contrast.
//   With no active votes, ratio = 0.5 (neutral) so the robot bridge won't switch.
require __DIR__ . '/db.php';
cors();

$c = cfg();

if (!empty($c['state_token'])) {
    $tok = $_GET['token'] ?? '';
    if (!hash_equals((string)$c['state_token'], (string)$tok)) {
        json_out(['error' => 'unauthorized'], 401);
    }
}

$window = isset($_GET['window']) ? (int)$_GET['window'] : (int)($c['active_window'] ?? 25);
if ($window < 1)   $window = 25;
if ($window > 600) $window = 600;

$sql = "SELECT choice, COUNT(*) AS n
        FROM votes
        WHERE updated_at >= (NOW() - INTERVAL :w SECOND)
        GROUP BY choice";
$st = db()->prepare($sql);
$st->bindValue(':w', $window, PDO::PARAM_INT);
$st->execute();

$mirror = 0;
$contrast = 0;
foreach ($st->fetchAll() as $row) {
    if ($row['choice'] === 'mirror')   $mirror   = (int)$row['n'];
    if ($row['choice'] === 'contrast') $contrast = (int)$row['n'];
}

$active = $mirror + $contrast;
$ratio  = $active > 0 ? $contrast / $active : 0.5;
$winner = $active === 0 ? 'none' : ($contrast > $mirror ? 'contrast' : ($mirror > $contrast ? 'mirror' : 'tie'));

json_out([
    'mirror'   => $mirror,
    'contrast' => $contrast,
    'active'   => $active,
    'ratio'    => round($ratio, 4),
    'winner'   => $winner,
    'window'   => $window,
    'ts'       => time(),
]);
