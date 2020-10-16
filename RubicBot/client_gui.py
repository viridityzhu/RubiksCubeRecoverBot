from tkinter import *
import socket
import face
import cubie
import arduino
import colorRecHSV
import recognize as Rec

# ------------------------常量和变量-------------------
DEFAULT_HOST = 'localhost'
DEFAULT_PORT = '8080'
DEFAULT_COM = '/dev/cu.usbserial-14530'
width = 60  # width of a facelet in pixels
facelet_id = [[[0 for col in range(3)]
               for row in range(3)] for face in range(6)]
colorpick_id = [0 for i in range(6)]
curcol = None
# 上黄右橘前绿下白左红后蓝
t = ("U", "R", "F", "D", "L", "B")
cols = ("yellow", "orange", "green", "white", "red", "blue")
# ----------------------------------------------------

# --------------------functions------------------------


def show_text(txt):
    print(txt)
    display.insert(INSERT, txt)
    root.update_idletasks()


def create_facelet_rects(a):
    """
             5L
        1U   3F   4D
             2R
             6B
    """
    offset = ((1, 0), (2, 1), (1, 1), (1, 2), (0, 1), (3, 1))
    for f in range(6):
        for row in range(3):
            y = 10 + offset[f][1] * 3 * a + row * a
            for col in range(3):
                x = 10 + offset[f][0] * 3 * a + col * a
                # 第f面第row行第col列
                facelet_id[f][row][col] = canvas.create_rectangle(
                    x, y, x + a, y + a, fill="grey")
                if row == 1 and col == 1:
                    canvas.create_text(
                        x + width // 2, y + width // 2, font=("", 14), text=t[f], state=DISABLED)
    for f in range(6):
        canvas.itemconfig(facelet_id[f][1][1], fill=cols[f])


def create_colorpick_rects(a):
    global curcol
    global cols
    for i in range(6):
        x = (i % 3) * (a + 5) + 7 * a
        y = (i // 3) * (a + 5) + 7 * a
        colorpick_id[i] = canvas.create_rectangle(
            x, y, x + a, y + a, fill=cols[i])
        canvas.itemconfig(colorpick_id[0], width=4)
        curcol = cols[0]


def get_definition_string():
    """
        魔方状态定义字符串：用颜色对应的面来代替颜色进行编码
    """
    color_to_facelet = {}
    for i in range(6):
        color_to_facelet.update(
            {canvas.itemcget(facelet_id[i][1][1], "fill"): t[i]})  # {中心块颜色:块对应的面, e.g. yellow: U}
    s = ''
    for f in range(6):
        for row in range(3):
            for col in range(3):
                s += color_to_facelet[canvas.itemcget(
                    facelet_id[f][row][col], "fill")]
    return s
# ----------------------------------------------------

# --------------------------求解--------------------------


def solve():
    display.delete(1.0, END)
    # -----------------------连接server------------------------
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    except socket.error:
        show_text('Failed to create socket')
        return
    host = txt_host.get(1.0, END).rstrip()  # default: localhost
    port = int(txt_port.get(1.0, END))  # default: 8080

    try:
        remote_ip = socket.gethostbyname(host)
    except socket.gaierror:
        show_text('Hostname could not be resolved.')
        return
    try:
        s.connect((remote_ip, port))
    except:
        show_text('Cannot connect to server!')
        return
    show_text('Connected with ' + remote_ip + '\n')
    # -----------------------------------------------
    # --------------------获取魔方状态编码---------------------------
    try:
        defstr = get_definition_string() + '\n'
    except:
        show_text('Invalid facelet configuration.\nWrong or missing colors.')
        return
    show_text(defstr)
    # -----------------------------------------------
    # ----------------------发送到服务器，获得解-------------------------
    try:
        s.sendall((defstr + '\n').encode())
    except:
        show_text('Cannot send cube configuration to server.')
        return
    solve_step = s.recv(2048).decode()  # 带空格的解
    show_text(solve_step)
    # -----------------------------------------------
    # --------------------将解发送给Arduino---------------------------
    solution = solve_step.replace(' ', '')  # 不带空格的解
    com = txt_com.get(1.0, END).rstrip()  # 串口
    arduino.send_arduino(com, solution)
    print(com, solution)
# ----------------------------------------------------

# -------------------------面的颜色---------------------------


def clean():
    for f in range(6):
        for row in range(3):
            for col in range(3):
                # 用中心块的颜色填每一块
                canvas.itemconfig(facelet_id[f][row][col], fill=canvas.itemcget(
                    facelet_id[f][1][1], "fill"))


def empty():
    for f in range(6):
        for row in range(3):
            for col in range(3):
                if row != 1 or col != 1:
                    # 全填灰色，除了中心块
                    canvas.itemconfig(facelet_id[f][row][col], fill="grey")


def random():
    cc = cubie.CubieCube()
    cc.randomize()
    fc = cc.to_facelet_cube()
    idx = 0
    for f in range(6):
        for row in range(3):
            for col in range(3):
                canvas.itemconfig(facelet_id[f][row][
                                  col], fill=cols[fc.f[idx]])
                idx += 1
# ----------------------------------------------------

# -------------------------编辑面的颜色---------------------------


def click(event):
    global curcol
    idlist = canvas.find_withtag("current")
    if len(idlist) > 0:
        if idlist[0] in colorpick_id:
            curcol = canvas.itemcget("current", "fill")
            for i in range(6):
                canvas.itemconfig(colorpick_id[i], width=1)
            canvas.itemconfig("current", width=5)
        else:
            canvas.itemconfig("current", fill=curcol)
# ----------------------------------------------------
# --------------------------调用recognize--------------------------


def recognize():
    """
    BOOWYGYWO
    BYGOOOOBY
    YRBOGWBBR
    YYWOWGWYW
    OGGORBOGO
    GOOWBYGBW

    BRRDUFUDR
    BUFRRRRBU
    ULBRFDBBL
    UUDRDFDUD
    RFFRLBRFR
    FRRDBUFBD
    t = ("U", "R", "F", "D", "L", "B")
    cols = ("yellow", "orange", "green", "white", "red", "blue")
    上黄右橘前绿下白左红后蓝
    """
    faceletToColor = {}
    for i in range(6):
        faceletToColor.update(
            {t[i]: canvas.itemcget(facelet_id[i][1][1], "fill")})  # {中心块颜色:块对应的面, e.g. yellow: U}
    isSuccess = Rec.rec()
    if(isSuccess):
        s = colorRecHSV.scancubemain()
        idx = 0
        for f in range(6):
            for row in range(3):
                for col in range(3):
                    canvas.itemconfig(facelet_id[f][row][
                                      col], fill=faceletToColor[s[idx]])
                    idx += 1


# -----------------------tkinter GUI-----------------------------
root = Tk()
root.wm_title("Solver Client")
canvas = Canvas(root, width=12 * width + 20, height=9 * width + 20)
canvas.pack()

brecognize = Button(text="Recognize", height=2, width=10,
                    relief=RAISED, command=recognize)
brecognize_window = canvas.create_window(
    10 + 0 * width, 10 + 6.5 * width, anchor=NW, window=brecognize)
bsolve = Button(text="Solve", height=2, width=10, relief=RAISED, command=solve)
bsolve_window = canvas.create_window(
    10 + 10.5 * width, 10 + 6.5 * width, anchor=NW, window=bsolve)
bclean = Button(text="Clean", height=1, width=10, relief=RAISED, command=clean)
bclean_window = canvas.create_window(
    10 + 10.5 * width, 10 + 7.5 * width, anchor=NW, window=bclean)
bempty = Button(text="Empty", height=1, width=10, relief=RAISED, command=empty)
bempty_window = canvas.create_window(
    10 + 10.5 * width, 10 + 8 * width, anchor=NW, window=bempty)
brandom = Button(text="Random", height=1, width=10,
                 relief=RAISED, command=random)

brandom_window = canvas.create_window(
    10 + 10.5 * width, 10 + 8.5 * width, anchor=NW, window=brandom)
display = Text(height=7, width=39)
text_window = canvas.create_window(
    10 + 6.5 * width, 10 + .5 * width, anchor=NW, window=display)
hp = Label(text='Host & Port & 串口')
hp_window = canvas.create_window(
    10 + 0 * width, 10 + 0.6 * width, anchor=NW, window=hp)
txt_host = Text(height=1, width=20)
txt_host_window = canvas.create_window(
    10 + 0 * width, 10 + 1 * width, anchor=NW, window=txt_host)
txt_host.insert(INSERT, DEFAULT_HOST)
txt_port = Text(height=1, width=20)
txt_port_window = canvas.create_window(
    10 + 0 * width, 10 + 1.5 * width, anchor=NW, window=txt_port)
txt_port.insert(INSERT, DEFAULT_PORT)
txt_com = Text(height=1, width=20)
txt_com_window = canvas.create_window(
    10 + 0 * width, 10 + 2 * width, anchor=NW, window=txt_com)
txt_com.insert(INSERT, DEFAULT_COM)
canvas.bind("<Button-1>", click)
create_facelet_rects(width)
create_colorpick_rects(width)

root.mainloop()
