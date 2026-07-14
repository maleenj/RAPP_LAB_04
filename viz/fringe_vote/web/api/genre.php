<?php
// GET  -> {genre, ts}            current roulette selection (or null)
// POST {genre} -> {ok, genre}    set the selection (called by roulette.html on spin stop)
require __DIR__ . '/db.php';
cors();

$GENRES = ['superhero', 'jhorror', 'romcom', 'bollywood'];

if (($_SERVER['REQUEST_METHOD'] ?? 'GET') === 'POST') {
    $in = body_json();
    $g = $in['genre'] ?? ($_POST['genre'] ?? '');
    if (!in_array($g, $GENRES, true)) {
        json_out(['error' => 'bad genre'], 400);
    }
    db()->prepare("UPDATE genre SET value = :v WHERE id = 1")->execute([':v' => $g]);
    json_out(['ok' => true, 'genre' => $g]);
}

$row = db()->query("SELECT value, UNIX_TIMESTAMP(updated_at) AS ts FROM genre WHERE id = 1")->fetch();
json_out([
    'genre' => $row['value'] ?? null,
    'ts'    => isset($row['ts']) ? (int)$row['ts'] : 0,
]);
