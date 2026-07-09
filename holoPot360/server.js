// sync-server.js
// Run with: node server.js
// Both phones open: http://<your-laptop-ip>:3000/?mode=statue32&phone=1
//                   http://<your-laptop-ip>:3000/?mode=statue32&phone=2
// Control page:     http://<your-laptop-ip>:3000/control

const http = require('http');
const fs   = require('fs');
const path = require('path');
const { WebSocketServer } = require('ws');

// -----------------------------------------------------------
// HTTP server — serves index.html and static files from ./
// -----------------------------------------------------------
const httpServer = http.createServer((req, res) => {
  let filePath = '.' + req.url.split('?')[0]; // strip query string
  if (filePath === './') filePath = './index.html';
  if (filePath === './control') filePath = './control.html';

  const ext = path.extname(filePath);
  const mime = {
    '.html': 'text/html',
    '.js':   'application/javascript',
    '.png':  'image/png',
    '.jpg':  'image/jpeg',
  }[ext] || 'application/octet-stream';

  fs.readFile(filePath, (err, data) => {
    if (err) { res.writeHead(404); res.end('Not found: ' + filePath); return; }
    res.writeHead(200, { 'Content-Type': mime });
    res.end(data);
  });
});

// -----------------------------------------------------------
// WebSocket server
// -----------------------------------------------------------
const wss = new WebSocketServer({ server: httpServer });

let clients = new Set();
let nudge = 0; // frame nudge for phone 2 (can go + or -)

wss.on('connection', (ws) => {
  clients.add(ws);
  console.log('Client connected. Total:', clients.size);

  ws.on('message', (raw) => {
    // Control page sends: { type: 'nudge', delta: +1 or -1 }
    // or { type: 'start' }
    try {
      const msg = JSON.parse(raw);

      if (msg.type === 'start') {
        // Broadcast the same epoch to ALL phones simultaneously
        const epoch = Date.now();
        const payload = JSON.stringify({ type: 'start', epoch, nudge });
        console.log('START broadcast — epoch:', epoch, 'nudge:', nudge);
        for (const c of clients) {
          if (c.readyState === 1) c.send(payload);
        }
      }

      if (msg.type === 'nudge') {
        nudge += msg.delta;
        const payload = JSON.stringify({ type: 'nudge', nudge });
        console.log('NUDGE — nudge now:', nudge);
        for (const c of clients) {
          if (c.readyState === 1) c.send(payload);
        }
      }

      if (msg.type === 'reset') {
        nudge = 0;
        const payload = JSON.stringify({ type: 'nudge', nudge });
        console.log('RESET nudge to 0');
        for (const c of clients) {
          if (c.readyState === 1) c.send(payload);
        }
      }
    } catch(e) { console.error('Bad message', e); }
  });

  ws.on('close', () => {
    clients.delete(ws);
    console.log('Client disconnected. Total:', clients.size);
  });
});

httpServer.listen(3000, '0.0.0.0', () => {
  console.log('');
  console.log('==============================================');
  console.log(' Hologram sync server running on port 3000');
  console.log('==============================================');
  console.log('');
  console.log(' Find your laptop IP with:');
  console.log('   Mac/Linux: ifconfig | grep "inet "');
  console.log('   Windows:   ipconfig');
  console.log('');
  console.log(' Then open on each phone:');
  console.log('   http://<laptop-ip>:3000/?mode=statue32&phone=1');
  console.log('   http://<laptop-ip>:3000/?mode=statue32&phone=2');
  console.log('');
  console.log(' Control page (laptop browser):');
  console.log('   http://localhost:3000/control');
  console.log('');
});
