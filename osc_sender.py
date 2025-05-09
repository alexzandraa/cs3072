from pythonosc import udp_client

# Set up a client to send messages to localhost on port 5000
client = udp_client.SimpleUDPClient("127.0.0.1", 5000)

# Send a test message 
client.send_message("/muse/eeg", [1, 2, 3, 4])
print("Test OSC message sent")
