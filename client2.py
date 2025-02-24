# this code is for local client
import os

ll, ul = 4000, 8000
import socket
from time import sleep

# create folder in cwd with date and time as name
os.chdir("outputs")


MCAST_GRP = '224.1.1.1'
MCAST_PORT = 5007
MULTICAST_TTL = 2

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, MULTICAST_TTL)

first_iter = True
model_from_server = None
client_no = 2
password = client_no
round_no = 0
while True:
    if first_iter:
        import basehash
        hash_fn = basehash.base36()
        userid_hashed = hash_fn.hash(client_no)
        password_hashed = hash_fn.hash(password)
        import pickle
        hashed_arr = pickle.dumps([userid_hashed, password_hashed])
        sock.sendto(bytes(hashed_arr), (MCAST_GRP, MCAST_PORT))
        first_iter = False
    else:
        ll = ul
        ul += 100

        from cryptography.fernet import Fernet
        import pickle

        encryption_arr = pickle.loads(model_from_server)
        model_from_server = encryption_arr[1]
        model_from_server = Fernet(encryption_arr[0]).decrypt(model_from_server)
        model_from_server = pickle.loads(model_from_server)

        if round_no == 0:
            weights, biases, metrics = model_from_server.myMLP(model_from_server, [round_no, client_no, ll, ul])
        else:
            weights, biases, metrics = model_from_server.myMLP([round_no, client_no, ll, ul])

        # send the features to the server
        data = pickle.dumps([weights, biases, metrics])
        from cryptography.fernet import Fernet

        key = Fernet.generate_key()
        f = Fernet(key)
        encrypted_data = f.encrypt(data)
        encryption_arr = pickle.dumps([key, encrypted_data])
        sock.sendto(bytes(encryption_arr), (MCAST_GRP, MCAST_PORT))
        print("Features sent, waiting for the model...  ", end="")
        round_no += 1

    # wait for the global server to send a message
    model_from_server = sock.recv(50000000)
    if isinstance(model_from_server, str):
        print("Training completed!")
        exit(0)
    print("New model received!")
    sleep(2)
