// Client example using Node.js and socket.io-client
const io = require("socket.io-client");
const socket = io("http://localhost:3000");

socket.on("connect", () => {
  console.log("Connected to server");
  socket.emit("startSimulation", { simulation: "start" });
});

socket.on("disconnect", () => {
  console.log("Disconnected from server");
});