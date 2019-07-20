import numpy as np
import cv2
import pygame
import tkinter as tk
import time
from cuda_fractal import compute_newton

A = .5 + .5j
P = [-1, 0, 0, 1]
N_ITERATIONS = 120
RED = 0
GREEN = 150
BLUE = 255

RATIO = 16 / 9

WIDTH = 1440
HEIGHT = int(WIDTH / RATIO)
X_RAD = 3
Y_RAD = X_RAD / RATIO
X_CENTER = 0
Y_CENTER = 0

ZOOM_FACTOR = 2

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Newton Display")
fractal = pygame.Surface(screen.get_size())

class Newton:

    def __init__(self, a, p, width, height, x_rad, y_rad, x_center, y_center, red, green, blue, n_iterations, zoom_factor):
        self.a = a
        self.p = p
        self.save_count = 0
        self.width = width
        self.height = height
        self.x_rad = x_rad
        self.y_rad = y_rad
        self.x_center = x_center
        self.y_center = y_center
        self.red = red
        self.green = green
        self.blue = blue
        self.n_iterations = n_iterations
        self.zoom_factor = zoom_factor
        x, y = self.grid()
        self.fractal = self.compute_fractal(x, y)
        
    def grid(self, save = False):
        return (np.linspace(-self.x_rad + self.x_center, self.x_rad + self.x_center, self.width), 
            np.linspace(-self.y_rad - self.y_center, self.y_rad - self.y_center, self.height))
    
    def compute_fractal(self, x, y):
        cs = compute_newton(x, y, self.n_iterations, np.asarray(self.p), self.a)
        return cs
        
    def color_fractal(self):
        cs = np.reshape(self.fractal, (self.fractal.shape[0], self.fractal.shape[1], 1))
        cols = np.tile([self.red, self.green, self.blue], (self.fractal.shape[0], self.fractal.shape[1], 1))
        img = (cs * cols).astype(np.uint8)
        return img
        
    def save(self):
        img = self.color_fractal()
        name = "Newton" + "_" 
            + str(self.a) + "_" 
            + str(self.p) + "_" 
            + str(self.red) + "_" 
            + str(self.green) + "_" 
            + str(self.blue) + "_N" 
            + str(self.n_iterations) + "_" 
            + str(self.save_count) + ".jpg"
        cv2.imwrite(name, np.transpose(img[:, :, ::-1], (1, 0, 2)))
        self.save_count += 1
        print("Image saved as '{}'".format(name))
        
    def translate_fractal_x(self, x_pix_trans): # could be optimized to translate x and y at the same time
        self.x_center +=  self.x_rad * x_pix_trans / (self.width // 2)
        if x_pix_trans > 0:
            old_img = self.fractal[x_pix_trans:, :]
            x, y = self.grid()
            x = x[-x_pix_trans:]
            x_new_img = self.compute_fractal(x, y)
            return np.concatenate([old_img, x_new_img], axis = 0)
        elif x_pix_trans < 0:
            old_img = self.fractal[:x_pix_trans, :]
            x, y = self.grid()
            x = x[:-x_pix_trans]
            x_new_img = self.compute_fractal(x, y)
            return np.concatenate([x_new_img, old_img], axis = 0)
        elif x_pix_trans == 0:
            return self.fractal
            
    def translate_fractal_y(self, y_pix_trans):
        self.y_center -= self.y_rad * y_pix_trans / (self.height // 2)
        if y_pix_trans > 0:
            old_img = self.fractal[:, y_pix_trans:]
            x, y = self.grid()
            y = y[-y_pix_trans:]
            y_new_img = self.compute_fractal(x, y)
            return np.concatenate([old_img, y_new_img], axis = 1)
        elif y_pix_trans < 0:
            old_img = self.fractal[:, :y_pix_trans]
            x, y = self.grid()
            y = y[:-y_pix_trans]
            y_new_img = self.compute_fractal(x, y)
            return np.concatenate([y_new_img, old_img], axis = 1)
        elif y_pix_trans == 0:
            return self.fractal

    def event_handler(self):
        keys = pygame.key.get_pressed()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == pygame.MOUSEBUTTONDOWN:
                self.fractal = self.translate_fractal_x(pygame.mouse.get_pos()[0] - self.width // 2)
                self.fractal = self.translate_fractal_y(pygame.mouse.get_pos()[1] - self.height // 2)
                display_fractal(self.color_fractal())
        
newton = Newton(A, P, WIDTH, HEIGHT, X_RAD, Y_RAD, X_CENTER, Y_CENTER, RED, GREEN, BLUE, N_ITERATIONS, ZOOM_FACTOR)

def display_fractal(img):
    pygame.surfarray.blit_array(fractal, img)
    screen.blit(fractal, (0, 0))
    pygame.display.update()

root = tk.Tk()
root.title("Newton Controls")
root.geometry("530x180")

def click_update(event = None):
    newton.red = int(E_red.get())
    newton.green = int(E_green.get())
    newton.blue = int(E_blue.get())
    n_iter = newton.n_iterations
    newton.n_iterations = int(E_it.get())
    a = newton.a
    newton.a = complex(E_a.get())
    p = newton.p
    p_str = E_p.get()
    newton.p = [complex(c) for c in p_str.split(',')]
    if n_iter != newton.n_iterations or a != newton.a or p != newton.p: 
        x, y = newton.grid()
        newton.fractal = newton.compute_fractal(x, y)
    display_fractal(newton.color_fractal())

def click_zoom_in(event = None):
    newton.zoom_factor = float(E_zoom.get())
    newton.x_rad /= newton.zoom_factor
    newton.y_rad /= newton.zoom_factor
    x, y = newton.grid()
    newton.fractal = newton.compute_fractal(x, y)
    display_fractal(newton.color_fractal())
    
def click_zoom_out(event = None):
    newton.zoom_factor = float(E_zoom.get())
    newton.x_rad *= newton.zoom_factor
    newton.y_rad *= newton.zoom_factor
    x, y = newton.grid()
    newton.fractal = newton.compute_fractal(x, y)
    display_fractal(newton.color_fractal())
    
def click_save(event = None):
    newton.save()

def click_recenter(event = None):
    newton.x_center = X_CENTER
    newton.y_center = Y_CENTER
    x, y = newton.grid()
    newton.fractal = newton.compute_fractal(x, y)
    display_fractal(newton.color_fractal())
    
def click_recenter_unzoom(event = None):
    newton.x_center = X_CENTER
    newton.y_center = Y_CENTER
    newton.x_rad = X_RAD
    newton.y_rad = Y_RAD
    x, y = newton.grid()
    newton.fractal = newton.compute_fractal(x, y)
    display_fractal(newton.color_fractal())
    
window = tk.Frame(root)
window.grid()

L_red = tk.Label(root, text = "Red")
L_red.grid(column = 0, row = 0)
E_red = tk.Entry(root, bd = 5, width = 4)
E_red.insert(tk.END, str(RED))
E_red.grid(column = 1, row = 0)
L_green = tk.Label(root, text = "Green")
L_green.grid(column = 0, row = 1)
E_green = tk.Entry(root, bd = 5, width = 4)
E_green.grid(column = 1, row = 1)
E_green.insert(tk.END, str(GREEN))
L_blue = tk.Label(root, text = "Blue")
L_blue.grid(column = 0, row = 2)
E_blue = tk.Entry(root, bd = 5, width = 4)
E_blue.grid(column = 1, row = 2)
E_blue.insert(tk.END, str(BLUE))
L_it = tk.Label(root, text = "Iterations")
L_it.grid(column = 0, row = 3)
E_it = tk.Entry(root, bd = 5, width = 4)
E_it.grid(column = 1, row = 3)
E_it.insert(tk.END, str(N_ITERATIONS))
L_a = tk.Label(root, text = "A")
L_a.grid(column = 0, row = 4)
E_a = tk.Entry(root, bd = 5, width = 10)
E_a.grid(column = 1, row = 4)
E_a.insert(tk.END, str(A)[1:-1])
L_p = tk.Label(root, text = "Polynomial")
L_p.grid(column = 0, row = 5)
E_p = tk.Entry(root, bd = 5, width = 30)
E_p.grid(column = 1, row = 5)
E_p.insert(tk.END, str(P)[1:-1])
update = tk.Button(root, text = "Update", command = click_update)
update.grid(column = 1, row = 6)
L_zoom = tk.Label(root, text = "Zoom factor")
L_zoom.grid(column = 2, row = 0)
E_zoom = tk.Entry(root, bd = 5, width = 4)
E_zoom.grid(column = 3, row = 0)
E_zoom.insert(tk.END, str(ZOOM_FACTOR))
zoom_in = tk.Button(root, text = "Zoom in", command = click_zoom_in)
zoom_in.grid(column = 3, row = 1)
zoom_out = tk.Button(root, text = "Zoom out", command = click_zoom_out)
zoom_out.grid(column = 3, row = 2)
save = tk.Button(root, text = "Save", command = click_save)
save.grid(column = 3, row = 4)
recenter = tk.Button(root, text = "Recenter", command = click_recenter)
recenter.grid(column = 3, row = 5)
recenter_unzoom = tk.Button(root, text = "Recenter + Unzoom", command = click_recenter_unzoom)
recenter_unzoom.grid(column = 3, row = 6)

display_fractal(newton.color_fractal())
while True: 
    newton.event_handler()
    window.update()
    