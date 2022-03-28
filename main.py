import collections
import math
import os
import pickle
import zipfile
import numpy as np
import sys
from tkinter import *
from PIL import ImageTk, Image
from tkinter import filedialog, messagebox

np.set_printoptions(threshold=sys.maxsize)


class Node:
    def __init__(self, prob, symbol, left=None, right=None):
        # probability of symbol
        self.prob = prob

        # symbol
        self.symbol = symbol

        # left node
        self.left = left

        # right node
        self.right = right

        # tree direction (0/1)
        self.code = ''


codes = dict()


def Calculate_Codes(node, val=''):
    newVal = val + str(node.code)

    if node.left:
        Calculate_Codes(node.left, newVal)
    if node.right:
        Calculate_Codes(node.right, newVal)

    if not node.left and not node.right:
        codes[node.symbol] = newVal

    return codes


def Calculate_Probability(data):
    symbols_dict = dict()
    for element in data:
        if symbols_dict.get(element) is None:
            symbols_dict[element] = 1
        else:
            symbols_dict[element] += 1
    return symbols_dict


def Output_Encoded(data, coding):
    encoding_output = []
    for c in data:
        encoding_output.append(coding[c])

    string = ''.join([str(item) for item in encoding_output])
    return string


# lvl1
def encode_txt(data):
    text_file = open(data, "r")
    txt = text_file.read()
    text_file.close()

    symbol_with_probs = Calculate_Probability(txt)
    symbols = list(symbol_with_probs.keys())
    values = list(symbol_with_probs.values())
    nodes = []

    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.prob)

        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        newNode = Node(left.prob + right.prob, left.symbol + right.symbol, left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])
    encoded_output = Output_Encoded(txt, huffman_encoding)

    mytxt = open(r'encoded.txt', 'w')
    encoded_text = encoded_output
    mytxt.write(encoded_text)
    mytxt.close()

    # entropy for txt
    prob = []

    for i in range(len(values)):
        prob.append(values[i] / len(txt))
    H = 0
    for i in range(len(prob)):
        H = H + (prob[i] * math.log2(prob[i]))

    # calculations
    beforC = len(txt) * 8
    afterC = len(encoded_output)
    entropy = -H

    pickle.dump(nodes[0], open("tree.pkl", "wb"))

    list_files = ['encoded.txt', 'tree.pkl']

    with zipfile.ZipFile('text.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    for file in list_files:
        os.remove(file)

    return entropy, beforC, afterC


def decode_txt(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

    huffman_tree = pickle.load(open("tree.pkl", "rb"))
    tree_head = huffman_tree
    decoded_output = []
    text_file = open('encoded.txt', "r")
    encoded_data = text_file.read()
    text_file.close()

    for x in encoded_data:
        if x == '1':
            huffman_tree = huffman_tree.right
        elif x == '0':
            huffman_tree = huffman_tree.left
        if huffman_tree.left is None and huffman_tree.right is None:
            decoded_output.append(huffman_tree.symbol)
            huffman_tree = tree_head

    string = ''.join([str(item) for item in decoded_output])

    mytxt = open('decoded_txt', 'w')
    decoded_text = string
    mytxt.write(decoded_text)
    mytxt.close()

    list_files = ['encoded.txt', 'tree.pkl']
    for file in list_files:
        os.remove(file)

    return decoded_text


def Huffman_Encoding(data):
    symbol_with_probs = calculate_freq(data)
    symbols = symbol_with_probs.keys()

    nodes = []

    for symbol in symbols:
        nodes.append(Node(symbol_with_probs.get(symbol), symbol))

    while len(nodes) > 1:
        nodes = sorted(nodes, key=lambda x: x.prob)

        right = nodes[0]
        left = nodes[1]

        left.code = 0
        right.code = 1

        newNode = Node(int(left.prob + right.prob), int(left.symbol + right.symbol), left, right)

        nodes.remove(left)
        nodes.remove(right)
        nodes.append(newNode)

    huffman_encoding = Calculate_Codes(nodes[0])

    return huffman_encoding


def code_len(encode_data):
    x = encode_data
    code_length = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            code_length = code_length + len(x[i][j])

    return code_length


def rgb2gray(data):
    im = Image.open(data)
    im1 = Image.Image.split(im)

    r = np.array(im1[0])
    g = np.array(im1[1])
    b = np.array(im1[2])

    return r, g, b


def image2array(data):
    img = Image.open(data)
    imgarray = np.array(img.convert('L'))
    return imgarray


def calculate_freq(data):
    frequency = collections.Counter()
    x = data
    for sublist in x:
        frequency.update(sublist)

    return frequency


def encoder(data):
    tree = Huffman_Encoding(data)
    indexer = np.array([tree.get(i, -1) for i in range(data.min(), data.max() + 1)])
    encoded = indexer[(data - data.min())]

    return encoded, tree


def decoder(data, tree):
    d = tree
    asd = {value: key for key, value in d.items()}
    a = data

    keys_values = asd.items()
    new_d = {key: value for key, value in keys_values}
    b = np.empty_like(a)
    for old, new in new_d.items():
        b[a == old] = new
    b = b.astype('uint8')
    return b


def imageDiff(data):
    diff_arr = np.array(data, copy=True)

    N = data.shape[0]
    M = data.shape[1]
    for i in range(N):
        for j in range(1, M):
            second_column = int(data[i][j])
            first_column = int(data[i][j - 1])
            diff_arr[i][j] = second_column - first_column

    pivot = data[0][0]
    diff_arr[0][0] = data[0][0] - pivot
    for i in range(1, N):
        first_row = int(data[i - 1][0])
        second_row = int(data[i][0])
        diff_arr[i][0] = second_row - first_row

    return diff_arr, pivot


def diff2image(data, pivot):
    diff_arr = np.array(data, copy=True)
    diff_arr[0][0] = pivot
    N = data.shape[0]
    M = data.shape[1]

    for i in range(1, N):
        first_row = int(diff_arr[i - 1][0])
        second_row = int(diff_arr[i][0])
        diff_arr[i][0] = second_row + first_row

    for i in range(N):
        for j in range(1, M):
            second_column = int(diff_arr[i][j])
            first_column = int(diff_arr[i][j - 1])
            diff_arr[i][j] = second_column + first_column

    return diff_arr


def entropy(data):
    size = data.size
    prob = []
    frequency = collections.Counter()
    for sublist in data:
        frequency.update(sublist)
    values = list(frequency.values())
    for i in range(len(values)):
        prob.append(values[i] / size)
    H = 0
    for i in range(len(prob)):
        H = H + (prob[i] * math.log2(prob[i]))
    return -H


def rgb2gray(image):
    im = Image.open(image)
    im1 = Image.Image.split(im)

    r = np.array(im1[0])
    g = np.array(im1[1])
    b = np.array(im1[2])
    return r, g, b


# lvl2
def encode_gray(image):
    data = image2array(image)
    Entropy = entropy(data)
    encoded, tree = encoder(data)
    aComp = code_len(encoded)
    np.save('data.npy', encoded)
    pickle.dump(tree, open("tree.pkl", "wb"))

    list_files = ['data.npy', 'tree.pkl']

    with zipfile.ZipFile('gray_image.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    for file in list_files:
        os.remove(file)
    return Entropy, aComp


def decode_gray(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    data = np.load('data.npy')
    tree = pickle.load(open("tree.pkl", "rb"))
    data = decoder(data, tree)
    image = Image.fromarray(data)

    list_files = ['data.npy', 'tree.pkl']
    for file in list_files:
        os.remove(file)

    return image


# lvl3
def encode_gray_diff(image):
    global gray_pivot
    data, gray_pivot = imageDiff(image2array(image))
    Entropy = entropy(data)
    encoded, tree = encoder(data)
    aComp = code_len(encoded)
    np.save('data.npy', encoded)
    pickle.dump(tree, open("tree.pkl", "wb"))

    list_files = ['data.npy', 'tree.pkl']

    with zipfile.ZipFile('gray_difference.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    for file in list_files:
        os.remove(file)

    return Entropy, aComp


def decode_gray_diff(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()
    data = np.load('data.npy')
    tree = pickle.load(open("tree.pkl", "rb"))
    data = decoder(data, tree)
    im = diff2image(data, gray_pivot)
    image = Image.fromarray(im)

    list_files = ['data.npy', 'tree.pkl']
    for file in list_files:
        os.remove(file)

    return image


# lvl4
def encode_gray_levels(image):
    r, g, b = rgb2gray(image)
    Entropy = (entropy(r) + entropy(g) + entropy(b)) / 3
    r_encoded, r_tree = encoder(r)
    np.save('data_r.npy', r_encoded)
    pickle.dump(r_tree, open("tree_r.pkl", "wb"))

    g_encoded, g_tree = encoder(g)
    np.save('data_g.npy', g_encoded)
    pickle.dump(g_tree, open("tree_g.pkl", "wb"))

    b_encoded, b_tree = encoder(b)
    np.save('data_b.npy', b_encoded)
    pickle.dump(b_tree, open("tree_b.pkl", "wb"))

    aComp = (code_len(r_encoded) + code_len(g_encoded) + code_len(b_encoded)) / 3

    list_files = ['data_r.npy', 'data_g.npy', 'data_b.npy', 'tree_r.pkl', 'tree_g.pkl', 'tree_b.pkl']

    with zipfile.ZipFile('colored.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    for file in list_files:
        os.remove(file)

    return Entropy, aComp


def decode_gray_levels(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

    data_r = np.load('data_r.npy')
    tree_r = pickle.load(open("tree_r.pkl", "rb"))
    R = decoder(data_r, tree_r)

    data_g = np.load('data_g.npy')
    tree_g = pickle.load(open("tree_g.pkl", "rb"))
    G = decoder(data_g, tree_g)

    data_b = np.load('data_b.npy')
    tree_b = pickle.load(open("tree_b.pkl", "rb"))
    B = decoder(data_b, tree_b)

    decompressed_image = np.dstack((R, G, B))
    image = Image.fromarray(np.uint8(decompressed_image))
    list_files = ['data_r.npy', 'data_g.npy', 'data_b.npy', 'tree_r.pkl', 'tree_g.pkl', 'tree_b.pkl']
    for file in list_files:
        os.remove(file)

    return image


# lvl5
def encode_difference(image):
    global r_pivot
    global g_pivot
    global b_pivot

    r, g, b = rgb2gray(image)
    r, r_pivot = imageDiff(r)
    r_encoded, r_tree = encoder(r)
    np.save('data_r.npy', r_encoded)
    pickle.dump(r_tree, open("tree_r.pkl", "wb"))

    g, g_pivot = imageDiff(g)
    g_encoded, g_tree = encoder(g)
    np.save('data_g.npy', g_encoded)
    pickle.dump(g_tree, open("tree_g.pkl", "wb"))

    b, b_pivot = imageDiff(b)
    b_encoded, b_tree = encoder(b)
    np.save('data_b.npy', b_encoded)
    pickle.dump(b_tree, open("tree_b.pkl", "wb"))

    Entropy = (entropy(r) + entropy(g) + entropy(b)) / 3
    aComp = (code_len(r_encoded) + code_len(g_encoded) + code_len(b_encoded)) / 3

    list_files = ['data_r.npy', 'data_g.npy', 'data_b.npy', 'tree_r.pkl', 'tree_g.pkl', 'tree_b.pkl']

    with zipfile.ZipFile('colored_difference.zip', 'w') as zipF:
        for file in list_files:
            zipF.write(file, compress_type=zipfile.ZIP_DEFLATED)

    for file in list_files:
        os.remove(file)

    return Entropy, aComp


def decode_difference(zip_file):
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall()

    data_r = np.load('data_r.npy')
    tree_r = pickle.load(open("tree_r.pkl", "rb"))
    R = decoder(data_r, tree_r)

    data_g = np.load('data_g.npy')
    tree_g = pickle.load(open("tree_g.pkl", "rb"))
    G = decoder(data_g, tree_g)

    data_b = np.load('data_b.npy')
    tree_b = pickle.load(open("tree_b.pkl", "rb"))
    B = decoder(data_b, tree_b)

    decompressed_image = np.dstack((diff2image(R, r_pivot), diff2image(G, g_pivot), diff2image(B, b_pivot)))
    image = Image.fromarray(np.uint8(decompressed_image))

    list_files = ['data_r.npy', 'data_g.npy', 'data_b.npy', 'tree_r.pkl', 'tree_g.pkl', 'tree_b.pkl']
    for file in list_files:
        os.remove(file)

    return image


class App:

    def __init__(self):
        self.txt = None
        self.image = None
        self.Entropy = None
        self.bComp = None
        self.aComp = None
        self.ratio = None

        self.window = Tk()
        self.window.title('File Compressor')
        self.window.geometry("780x330")
        self.window.resizable(width=True, height=True)

        self.v = IntVar()
        Label(text="""          Choose a Method 
            for encoding or decoding:""").place(x=250, y=10)

        Radiobutton(text="Text File (Lvl 1)", variable=self.v, value=1).place(x=300, y=45)
        Radiobutton(text="Gray Image (Lvl 2)", variable=self.v, value=2).place(x=300, y=70)
        Radiobutton(text="Gray Difference (Lvl 3)", variable=self.v, value=3).place(x=300, y=95)
        Radiobutton(text="Colored (Lvl 4)", variable=self.v, value=4).place(x=300, y=120)
        Radiobutton(text="Colored Differences (Lvl 5)", variable=self.v, value=5).place(x=300, y=145)

        Button(self.window, text='Upload the File', command=self.viewer).place(x=90, y=250)
        Button(self.window, text='Encode the File', command=self.encode_method).place(x=310, y=190)
        Button(self.window, text='Upload the Encoded file', command=self.decode_method).place(x=290, y=230)



        self.window.update()
        self.window.mainloop()

    def restart(self):
        self.window.destroy()
        os.startfile("main.py")

    def open_file(self):
        filename = filedialog.askopenfilename(title='Select the image')
        return filename

    def viewer(self):
        x = self.v.get()
        if x == 1:
            txtarea = Text(self.window, width=30, height=15)
            tf = filedialog.askopenfilename(
                title="Open Text file",
                filetypes=(("Text Files", "*.txt"),)
            )
            self.txt = tf
            tf = open(tf)
            data = tf.read()
            tf.close()
            txtarea.insert(END, data)
            txtarea.place(x=2, y=2)

        else:
            try:
                filename = self.open_file()
                self.image = filename
                img = Image.open(filename)
                img = img.resize((250, 250), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                panel = Label(self.window, image=img)
                panel.image = img
                panel.grid(row=2)
            except:
                messagebox.showinfo(title="Error", message="For upload txt file, please select radio button 1.")

    def encode_method(self):
        x = self.v.get()
        if x == 1:
            self.Entropy, self.bComp, self.aComp = encode_txt(self.txt)
        elif x == 2:
            self.Entropy, self.aComp = encode_gray(self.image)
            self.bComp = os.path.getsize(self.image) * 8
        elif x == 3:
            self.Entropy, self.aComp = encode_gray_diff(self.image)
            self.bComp = os.path.getsize(self.image) * 8
        elif x == 4:
            self.Entropy, self.aComp = encode_gray_levels(self.image)
            self.bComp = os.path.getsize(self.image) * 8
        elif x == 5:
            self.Entropy, self.aComp = encode_difference(self.image)
            self.bComp = os.path.getsize(self.image) * 8

        self.ratio = self.bComp / self.aComp

        Label(self.window, text="   CALCULATIONS").place(x=500, y=10)
        Label(self.window, text="-Entropy: " + str(self.Entropy)).place(x=500, y=40)
        Label(self.window, text="-Before Compression: " + str(self.bComp)).place(x=500, y=70)
        Label(self.window, text="-After Compression: " + str(self.aComp)).place(x=500, y=100)
        Label(self.window, text="-Compression Ratio: " + str(self.ratio)).place(x=500, y=130)

    def decode_method(self):
        try:
            filename = filedialog.askopenfilename(
                title="Open Encoded file",
                filetypes=(("Zip Files", "*.zip"),)
            )

            x = self.v.get()

            if x == 1:
                text = decode_txt(filename)
                txtarea = Text(self.window, width=30, height=15)
                txtarea.insert(END, text)
                txtarea.place(x=480, y=0)

            else:
                if x == 2:
                    img = decode_gray(filename)
                elif x == 3:
                    img = decode_gray_diff(filename)
                elif x == 4:
                    img = decode_gray_levels(filename)
                elif x == 5:
                    img = decode_difference(filename)

                img = img.resize((250, 250), Image.ANTIALIAS)
                img = ImageTk.PhotoImage(img)
                panel = Label(self.window, image=img)
                panel.image = img
                panel.place(x=480, y=0)

        except:
            messagebox.showinfo(title="Error", message="Please select the correct decoder method.")


App()
