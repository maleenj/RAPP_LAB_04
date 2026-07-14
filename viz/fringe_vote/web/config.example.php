<?php
// Copy this file to config.php (same folder) and fill in your Hostinger MySQL details.
// Find these in hPanel -> Databases -> MySQL Databases after creating a database.
// config.php is git-ignored — never commit real credentials.
return [
    'db_host' => 'localhost',          // Hostinger shared hosting: almost always 'localhost'
    'db_name' => 'uXXXXXX_fringe',     // the database NAME hPanel shows (with the uXXXXXX_ prefix)
    'db_user' => 'uXXXXXX_voteuser',   // the database USERNAME
    'db_pass' => 'CHANGE_ME',          // the password you set

    // Seconds a vote stays "active" since its last update. Phones heartbeat ~10s,
    // so 25s tolerates a missed beat while still dropping phones that are put away.
    'active_window' => 25,

    // Optional shared secret. If non-empty, tally.php requires ?token=THIS to read.
    // Leave '' for an open show. If set, pass the same value to the robot bridge.
    'state_token' => '',
];
