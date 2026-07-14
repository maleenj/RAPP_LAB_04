/* vote-api.js — shared client for the Fringe voting backend.
 *
 * This is the CONTRACT between the visual pages and the PHP backend.
 * Any design (hand-built or generated) only needs to call these functions;
 * it never talks to the API directly.
 *
 *   VoteAPI.getClientId()            -> stable per-device id (localStorage)
 *   VoteAPI.sendVote('mirror'|'contrast')   -> Promise
 *   VoteAPI.onState(cb, ms)          -> poll state.php every ms; cb(state); returns stop()
 *   VoteAPI.getRatio()               -> last polled ratio (0=mirror .. 1=contrast); 0.5 until first poll
 *   VoteAPI.fetchState()             -> Promise<state>
 *   VoteAPI.pickGenre()              -> {key,label}; also POSTs the pick to genre.php
 *   VoteAPI.fetchGenre()             -> Promise<{genre, ts}>
 *   VoteAPI.GENRES                   -> [{key,label}, ...]
 *
 * state = { mirror, contrast, active, ratio, winner, window, ts }
 *
 * API base is auto-detected from this script's own URL, so the pages work whether
 * deployed at https://site.com/fringe/ or any other folder.
 */
(function (global) {
  'use strict';

  var scriptUrl = (document.currentScript && document.currentScript.src) || location.href;
  var API_BASE = new URL('./api/', scriptUrl).href;

  function uuid() {
    if (global.crypto && global.crypto.randomUUID) return global.crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function (ch) {
      var r = (Math.random() * 16) | 0;
      var v = ch === 'x' ? r : (r & 0x3) | 0x8;
      return v.toString(16);
    });
  }

  function getClientId() {
    var id = null;
    try { id = localStorage.getItem('fringe_client_id'); } catch (e) {}
    if (!id) {
      id = uuid();
      try { localStorage.setItem('fringe_client_id', id); } catch (e) {}
    }
    return id;
  }

  function sendVote(choice) {
    return fetch(API_BASE + 'vote.php', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ clientId: getClientId(), choice: choice }),
      keepalive: true
    }).then(function (r) { return r.json(); }).catch(function () { return { error: 'network' }; });
  }

  // NOTE: backend file is tally.php (NOT state.php — Hostinger's malware filter
  // blocks that generic filename). Keep this in sync with api/tally.php.
  function fetchState(windowSec) {
    var q = windowSec ? ('?window=' + windowSec) : '';
    return fetch(API_BASE + 'tally.php' + q).then(function (r) { return r.json(); });
  }

  var _ratio = 0.5;
  function getRatio() { return _ratio; }

  function onState(cb, ms) {
    ms = ms || 1000;
    var stopped = false;
    function tick() {
      if (stopped) return;
      fetchState().then(function (s) {
        if (s && typeof s.ratio === 'number') _ratio = s.ratio;
        if (cb) cb(s);
      }).catch(function () {}).then(function () {
        if (!stopped) setTimeout(tick, ms);
      });
    }
    tick();
    return function stop() { stopped = true; };
  }

  var GENRES = [
    { key: 'superhero', label: 'Hollywood Superhero' },
    { key: 'jhorror',   label: 'Japanese Horror' },
    { key: 'romcom',    label: 'English Rom-com' },
    { key: 'bollywood', label: 'Bollywood Musical' }
  ];

  function pickGenre() {
    var g = GENRES[(Math.random() * GENRES.length) | 0];
    fetch(API_BASE + 'genre.php', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ genre: g.key })
    }).catch(function () {});
    return g;
  }

  function fetchGenre() {
    return fetch(API_BASE + 'genre.php').then(function (r) { return r.json(); });
  }

  global.VoteAPI = {
    getClientId: getClientId,
    sendVote: sendVote,
    onState: onState,
    getRatio: getRatio,
    fetchState: fetchState,
    pickGenre: pickGenre,
    fetchGenre: fetchGenre,
    GENRES: GENRES,
    API_BASE: API_BASE
  };
})(window);
