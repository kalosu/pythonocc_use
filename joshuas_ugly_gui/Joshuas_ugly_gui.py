import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import time
import copy
import pandas as pd
from sklearn.cluster import SpectralClustering
from scipy.optimize import curve_fit
import csv
import tkinter as tk
from scipy.optimize import minimize
import cv2
from scipy.interpolate import griddata
from scipy.optimize import minimize_scalar
import datetime
import os

import workhorse as whs



def gui_a():
    print('GUIten Tag, what can I do for you today? :)')
    def method_one():
        print('.........')
        a=str(entry1.get().strip('"'))
        print(a)
        print(COSVAR.get())
        whs.plot_ply(a,COSVAR.get())

    def method_two():
        whs.twd_comp(1, whs.untilt_profile_byedges(whs.extr_prof_by_height(entry2.get())), 1)


    def method_three():
        if place:
            inpstr = entry3.get().strip()
            inpstr = inpstr.strip('"')
            inpstr = inpstr.replace('.txt', '') + '.ply'
            whs.txt_to_ply_simple(entry3.get().strip('"'), inpstr)
        if not place:
            whs.txt_to_ply_simple(entry3.get().strip('"'), entry3_2.get().strip('"'))

    def method_four():
        whs.base_extraction_a(entry4.get().strip('"'))

    def method_five():
        whs.here_iHough_again_a(entry5.get().strip('"'))


    def method_six():
        print(entry6.get().strip('"'))
        whs.plot_with_mag(entry6.get().strip('"'), entry6_3.get().strip('"'), entry6_2.get().strip('"'), entry6_1.get().strip('"'), COSVAR.get(),save.get())

    def method_seven():
        print(entry7.get().strip('"'))
        whs.cut(entry7.get().strip('"'),'jaegal.ply')

    def method_eight():
        print(entry8.get().strip('"'))
        whs.ply_to_txt(entry8_1.get().strip('"'),entry8.get().strip('"'))

    def method_nine():
        print(entry9.get().strip('"'))
        whs.redgrid(entry9_1.get().strip('"'),entry9.get().strip('"'),fit.get())

    def method_ten():
        print(entry10.get().strip('"'))
        whs.layervis_single(entry10.get().strip('"'),0.1)

    def method_eleven():
        print(entry11.get().strip('"'))
        whs.onegrid(entry11.get().strip('"'),entry11_1.get().strip('"'))

    def method_twelve():
        print(entry12.get().strip('"'))
        whs.icp_2forone(entry12.get().strip('"'),entry12_1.get().strip('"'))

    def method_thirteen():
        print(entry13.get().strip('"'))
        whs.plotheat(entry13.get().strip('"'))

    def method_fourteen():
        print(entry14.get().strip('"'))
        whs.cyrcle_mask(entry14.get().strip('"'), entry14_3.get().strip('"'), entry14_2.get().strip('"'), entry14_1.get().strip('"'))


    root = tk.Tk()
    root.geometry("600x800")
    root.title("How are you GUIng")

    frame1 = tk.Frame(root)
    frame1.pack()

    frame2 = tk.Frame(root)
    frame2.pack()

    frame3 = tk.Frame(root)
    frame3.pack()

    frame4 = tk.Frame(root)
    frame4.pack()

    frame5 = tk.Frame(root)
    frame5.pack()

    frame6 = tk.Frame(root)
    frame6.pack()

    frame7 = tk.Frame(root)
    frame7.pack()

    frame8 = tk.Frame(root)
    frame8.pack()

    frame9 = tk.Frame(root)
    frame9.pack()

    frame10 = tk.Frame(root)
    frame10.pack()

    frame11 = tk.Frame(root)
    frame11.pack()

    frame12 = tk.Frame(root)
    frame12.pack()

    frame13 = tk.Frame(root)
    frame13.pack()

    frame14 = tk.Frame(root)
    frame14.pack()

    button1 = tk.Button(frame1, text="1. Plot .ply File", command=method_one)
    button1.pack(side=tk.LEFT)

    COSVAR = tk.BooleanVar()
    checkbox = tk.Checkbutton(frame1, text="With COS", variable=COSVAR)
    checkbox.pack(side=tk.RIGHT)

    button2 = tk.Button(frame2, text="2. Target/measure Comp profile", command=method_two)
    button2.pack(side=tk.LEFT)

    place = tk.BooleanVar()
    checkbox3 = tk.Checkbutton(frame3, text="same name, place", variable=place)
    checkbox3.pack(side=tk.RIGHT)
    button3 = tk.Button(frame3, text="3. txt to ply", command=method_three)
    button3.pack(side=tk.LEFT)

    button4 = tk.Button(frame4, text="4. extract base", command=method_four)
    button4.pack(side=tk.LEFT)

    button5 = tk.Button(frame5, text="5. hough transform for center detection", command=method_five)
    button5.pack(side=tk.LEFT)

    button6 = tk.Button(frame6, text="6. plot xyz scaled", command=method_six)
    save = tk.BooleanVar()
    checkbox4 = tk.Checkbutton(frame6, text="save pcd", variable=save)
    checkbox4.pack(side=tk.RIGHT)
    button6.pack(side=tk.LEFT)

    button7 = tk.Button(frame7, text="7. manual cut of ply", command=method_seven)
    button7.pack(side=tk.LEFT)

    button8 = tk.Button(frame8, text="8. ply to .xyz", command=method_eight)
    button8.pack(side=tk.LEFT)

    button9 = tk.Button(frame9, text="9. xyz comp common gr", command=method_nine)
    button9.pack(side=tk.LEFT)

    button10 = tk.Button(frame10, text="10. vir profile cuts", command=method_ten)
    button10.pack(side=tk.LEFT)

    button11 = tk.Button(frame11, text="11. interpolate on grid", command=method_eleven)
    button11.pack(side=tk.LEFT)

    button12 = tk.Button(frame12, text="12. Leveling on xy plane", command=method_twelve)
    button12.pack(side=tk.LEFT)

    button13 = tk.Button(frame13, text="13. Heatmap", command=method_thirteen)
    button13.pack(side=tk.LEFT)

    checkbox = tk.Checkbutton(frame6, text="With COS", variable=COSVAR)
    checkbox.pack(side=tk.RIGHT)

    button14 = tk.Button(frame14, text="14. circle mask", command=method_fourteen)
    button14.pack(side=tk.LEFT)

    fit = tk.BooleanVar()
    #fit=False
    checkbox2 = tk.Checkbutton(frame9, text="multisample ICP fit", variable=fit)
    checkbox2.pack(side=tk.RIGHT)

    entry1 = tk.Entry(frame1)
    entry1.pack(side=tk.RIGHT)

    entry2 = tk.Entry(frame2)
    entry2.pack(side=tk.RIGHT)


    entry3_2 = tk.Entry(frame3)
    entry3_2.pack(side=tk.RIGHT)
    entry3 = tk.Entry(frame3)
    entry3.pack(side=tk.RIGHT)

    entry4 = tk.Entry(frame4)
    entry4.pack(side=tk.RIGHT)

    entry5 = tk.Entry(frame5)
    entry5.pack(side=tk.RIGHT)

    entry6 = tk.Entry(frame6)
    entry6.pack(side=tk.RIGHT)
    entry6_1 = tk.Entry(frame6,width=5)
    entry6_1.pack(side=tk.RIGHT)
    entry6_2 = tk.Entry(frame6,width=5)
    entry6_2.pack(side=tk.RIGHT)
    entry6_3 = tk.Entry(frame6, width=5)
    entry6_3.pack(side=tk.RIGHT)

    entry7 = tk.Entry(frame7)
    entry7.pack(side=tk.RIGHT)

    entry8 = tk.Entry(frame8)
    entry8.pack(side=tk.RIGHT)
    entry8_1 = tk.Entry(frame8)
    entry8_1.pack(side=tk.RIGHT)

    entry9 = tk.Entry(frame9)
    entry9.pack(side=tk.RIGHT)
    entry9_1 = tk.Entry(frame9)
    entry9_1.pack(side=tk.RIGHT)

    entry10 = tk.Entry(frame10)
    entry10.pack(side=tk.RIGHT)

    entry11 = tk.Entry(frame11)
    entry11.pack(side=tk.RIGHT)
    entry11_1 = tk.Entry(frame11, width=5)
    entry11_1.pack(side=tk.RIGHT)

    entry12 = tk.Entry(frame12, width=20)
    entry12.pack(side=tk.RIGHT)
    entry12_1 = tk.Entry(frame12, width=20)
    entry12_1.pack(side=tk.RIGHT)

    entry13 = tk.Entry(frame13, width=20)
    entry13.pack(side=tk.RIGHT)

    entry14 = tk.Entry(frame14)
    entry14.pack(side=tk.RIGHT)
    entry14_1 = tk.Entry(frame14,width=5)
    entry14_1.pack(side=tk.RIGHT)
    entry14_2 = tk.Entry(frame14,width=5)
    entry14_2.pack(side=tk.RIGHT)
    entry14_3 = tk.Entry(frame14, width=5)
    entry14_3.pack(side=tk.RIGHT)


    label = tk.Label(root, text="1. Reads a .ply file and plots it with o3d methode (also works for .xyz) \n 2. Compares a measured profile with function \n 3. Converts a .txt file into a .ply file ([input][output]) \n 4. extracts the lowest base from a .txt of 3d ponitcloud "
                                "\n 4. reads a 3d pointcloud from txt, extracts base (ponts on substrate). \n and finds center of structure with hough transform \n returns x,y coordinate of circle center\n 5. hough stuff \n 6. plots 3d pcd with scaling factor for x,y,z (path,x,y,z)"
                                "\n 7.type in a patch for ply, press k to lock camera, crt+ click to set mask, press c to cut mask, \n!!WHEN SAVING FILE DONT FORGET .ply AT END!!1!!111!, \n 8. this method also does stuff \n 9. two sets of measurement data are interpaolated on the same grid and substracted "
                                "\n 10. cut through surface and display profile \n 11. interpolate on meshgrid with given pitch (1. pitch, path) \n 12. Input pcd and level area of pcd. Pcd will be leveled using icp \n 13. is halt ne heatmap ne, click to see z vals\n 14. masks a circle variables x,ypos, rad")

    label.pack(side=tk.BOTTOM)

    root.mainloop()



if __name__ == "__main__":
    gui_a()