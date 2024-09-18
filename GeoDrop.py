from tkinter import *
import tkinter.filedialog as filedialog
import tkinter.messagebox as messagebox
import numpy as np
import math
from PIL import Image, ImageTk
import cv2
import os
from matplotlib import pyplot

frames = []
background = None
ROI = [0, 0, 0, 0]
pix_ratio = 376
threshold_num = 30
fitting_result_radius = []
fitting_result_volume = []
fitting_result_area = []
v_size = (0, 0)
project_path = ''

def process_img(img):
    global pix_ratio, threshold_num
    x0_cut = ROI[0]
    y0_cut = ROI[1]
    x1_cut = ROI[2]
    y1_cut = ROI[3]

    img_ROI = img[y0_cut - 1:y1_cut + 1, x0_cut:x1_cut]
    bg_ROI = background[y0_cut - 1:y1_cut + 1, x0_cut:x1_cut]
    img_grey = cv2.cvtColor(img_ROI, cv2.COLOR_BGR2GRAY)
    bg_grey = cv2.cvtColor(bg_ROI, cv2.COLOR_BGR2GRAY)
    img_sub = cv2.subtract(bg_grey, img_grey)
    th, img_th = cv2.threshold(img_sub, threshold_num, 255, cv2.THRESH_BINARY)

    img_ff = img_th.copy()
    h, w = img_th.shape[:2]
    maskUsed = np.zeros((h + 2, w + 2), np.uint8)

    # Floodfill from point (0, 0)
    cv2.floodFill(img_ff, maskUsed, (0, 0), 255)
    if img_ff[h - 2, 0] == 0:
        cv2.floodFill(img_ff, maskUsed, (0, h - 1), 255)
        print((0, h - 1))
    if img_ff[0, w - 2] == 0:
        cv2.floodFill(img_ff, maskUsed, (w - 1, 0), 255)
        print((w - 1, 0))
    if img_ff[h - 2, w - 2] == 0:
        cv2.floodFill(img_ff, maskUsed, (w - 1, h - 1), 255)
        print(w - 1, h - 1)

    # Combine the two images to get the foreground.
    im_out1 = img_th | cv2.bitwise_not(img_ff)

    kernelopen = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3), (-1, -1))
    im_out0 = cv2.morphologyEx(im_out1, cv2.MORPH_OPEN, kernelopen)

    # find contour
    contours, hierarchy = cv2.findContours(im_out0, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    return im_out0, contours

def analyze_radius(img):
    img1 = img.copy()
    x0_cut = ROI[0]
    y0_cut = ROI[1]
    x1_cut = ROI[2]
    y1_cut = ROI[3]
    im_out0, contours = process_img(img)
    if len(contours) < 1:
        return -1, img

    # draw contour
    cnt = contours[0]

    img_color1 = cv2.cvtColor(im_out0, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(img_color1, [cnt], 0, (0, 0, 255), 1)
    cnt_size = len(cnt)

    ltx = []

    for i in range(cnt_size):
        Q = cnt[i]
        q = Q[0][0]
        ltx.append(q)

    MINX_index = ltx.index(min(ltx))

    e = cnt[MINX_index][0]

    ex = e[0]
    ey = e[1]

    high = 170
    wide = 100
    x0 = int(ex - wide / 2)
    y0 = int(ey - high / 2)
    x1 = int(ex + wide / 2)
    y1 = int(ey + high / 2)

    # find fitting points
    ltX = []
    ltY = []
    list_X = []
    for i in range(cnt_size):
        Q = cnt[i]
        p = Q[0]
        list_X.append(float(p[0]))
        if x0 <= p[0] <= x1 and y0 <= p[1] <= y1:
            ltX.append(float(p[0]))
            ltY.append(float(p[1]))

    # fitting
    def circle_fitting(x, y):
        sumx = sum(x)  # ∑xi
        sumy = sum(y)  # ∑yi
        sumx2 = sum([ix ** 2 for ix in x])  # ∑(xi)**2
        sumy2 = sum([iy ** 2 for iy in y])  # ∑(yi)**2
        sumxy = sum([ix * iy for (ix, iy) in zip(x, y)])  # ∑(xi*yi)

        F = np.array([[sumx2, sumxy, sumx],
                      [sumxy, sumy2, sumy],
                      [sumx, sumy, len(x)]])

        G = np.array([[-sum([ix ** 3 + ix * iy ** 2 for (ix, iy) in zip(x, y)])],
                      [-sum([ix ** 2 * iy + iy ** 3 for (ix, iy) in zip(x, y)])],
                      [-sum([ix ** 2 + iy ** 2 for (ix, iy) in zip(x, y)])]])

        T = np.linalg.inv(F).dot(G)  

        cex = float(T[0] / -2)  
        cey = float(T[1] / -2)  
        re = math.sqrt(cex ** 2 + cey ** 2 - T[2])  # Fit radius
        error = sum([np.hypot(cex - ix, cey - iy) - re for (ix, iy) in zip(x, y)])  # error
        return cex, cey, re, error

    Xi = np.array(ltX)
    Yi = np.array(ltY)

    try:
        cex, cey, re, error = circle_fitting(Xi, Yi)

        print(cex, cey, re, error / re ** 2, end="")

        # filter irrational fitting result

        if re / pix_ratio > 1 or re / pix_ratio < 0.01:
            raise ValueError("Irrational Ratio number")

        if abs(error) / re ** 2 > 0.0001:
            raise ValueError("Fitting Error Exceeded")

        if x0_cut + max(list_X) + 10 < x1_cut:
            raise ValueError("Fitting location irrational")

    except np.linalg.LinAlgError:
        return -1, img1

    except ValueError:
        print("(x)")
        cv2.circle(img, (int(cex + x0_cut), int(cey + y0_cut)), int(re), (0, 0, 0), 2)
        cv2.rectangle(img, (x0 + x0_cut, y0 + y0_cut), (x1 + x0_cut, y1 + y0_cut), (255, 0, 0))
        cv2.circle(img, (int(cex + x0_cut), int(cey + y0_cut)), 1, (0, 0, 0), 3)
        return -1, img

    else:
        print("\n", end="")
        cv2.circle(img, (int(cex + x0_cut), int(cey + y0_cut)), int(re), (0, 255, 0), 2)
        cv2.rectangle(img, (x0 + x0_cut, y0 + y0_cut), (x1 + x0_cut, y1 + y0_cut), (255, 0, 0))
        cv2.circle(img, (int(cex + x0_cut), int(cey + y0_cut)), 1, (0, 0, 255), 3)

        return re / pix_ratio, img

def analyze_area_volume(img):
    img1 = img.copy()
    x0_cut = ROI[0]
    y0_cut = ROI[1]
    x1_cut = ROI[2]
    y1_cut = ROI[3]
    im_out0, contours = process_img(img)

    if len(contours) < 1:
        return -1, -1, img

    # draw contour
    cnt = contours[0]

    img_color1 = cv2.cvtColor(im_out0, cv2.COLOR_GRAY2BGR)

    cv2.drawContours(img_color1, [cnt], 0, (0, 0, 255), 1)
    cnt_size = len(cnt)

    size = img_color1.shape
    Ymax = size[1]
    Xmax = size[0]

    imgFix = np.zeros(size, np.uint8)

    # calculate volume _ integral method
    try:
        ltx = []

        for i in range(cnt_size):
            Q = cnt[i]
            x = Q[0][1]
            y = Q[0][0]
            if y == (Ymax - 1):
                ltx.append(x)

        xmin = min(ltx)
        xmax = max(ltx)
        xmid = (xmin + xmax) // 2

        ltXY_up = []
        ltXY_down = []
        ltX_up = []
        ltX_down = []

        for i in range(cnt_size):
            N = cnt[i]
            Y = N[0][0]
            X = N[0][1]
            if X < xmid and Y != Ymax:
                ltXY_up.append((Y, X))
                ltX_up.append(X)
            if X > xmid and Y != Ymax:
                ltXY_down.append((Y, X))
                ltX_down.append(X)

        for i in range(len(ltXY_up)):
            UP = ltXY_up[i]
            y = UP[0]
            x = UP[1]
            # print (x,y,M)
            imgFix[x, y] = (0, 0, 255)

        for i in range(len(ltXY_down)):
            DOWN = ltXY_down[i]
            y = DOWN[0]
            x = DOWN[1]
            # print (x,y,M)
            imgFix[x, y] = (0, 255, 0)

        for i in range(len(ltXY_down)):
            imgFix[[xmid]] = (255, 0, 0)

            # Sorting and filtering pixel points in up and down area respectively

        ltXY_up_1 = sorted(ltXY_up, key=lambda y_up: y_up[0])  # resort ltXY_up from ymin to ymax
        ltXY_down_1 = sorted(ltXY_down, key=lambda y_up: y_up[0])  # resort ltXY_down from ymin to ymax

        ltXY_up_2 = [ltXY_up_1[0]]
        ltXY_down_2 = [ltXY_down_1[0]]

        for i in range(len(ltXY_up_1) - 1):  # detele the pixel point with repeated x in up area
            UP1 = ltXY_up_1[i]
            UP2 = ltXY_up_1[i + 1]
            x1 = UP1[1]
            x2 = UP2[1]
            if x2 != x1:
                ltXY_up_2.append(UP2)

        for i in range(len(ltXY_down_1) - 1):  # detele the pixel point with repeated x in down area
            DOWN1 = ltXY_down_1[i]
            DOWN2 = ltXY_down_1[i + 1]
            x1 = DOWN1[1]
            x2 = DOWN2[1]
            if x2 != x1:
                ltXY_down_2.append(DOWN2)

        # Draw image after filtering pixel points

        imgFix1 = np.zeros(size, np.uint8)

        for i in range(len(ltXY_up_2)):
            UP = ltXY_up_2[i]
            y = UP[0]
            x = UP[1]
            # print (x,y,M)
            imgFix1[x, y] = (0, 0, 255)
            img[x + y0_cut, y + x0_cut] = (0, 0, 255)

        for i in range(len(ltXY_down_2)):
            DOWN = ltXY_down_2[i]
            y = DOWN[0]
            x = DOWN[1]
            # print (x,y,M)
            imgFix1[x, y] = (0, 255, 0)
            img[x + y0_cut, y + x0_cut] = (0, 255, 0)

        for i in range(Ymax):
            imgFix1[[xmid]] = (255, 0, 0)
            img[xmid + y0_cut] = (255, 0, 0)

        # caculate volume using processed pixel points

        volume_up = 0
        volume_down = 0
        area_up = 0
        area_down = 0

        for i in range(len(ltXY_up_2) - 1):
            up = ltXY_up_2[i]
            upup = ltXY_up_2[i + 1]
            x = up[1]
            y = up[0]
            xx = upup[1]
            yy = upup[0]
            x_mid = (x + xx) / 2
            R_up_1 = xmid - x
            R_up_2 = xmid - xx
            R_up_real_1 = R_up_1 / pix_ratio
            R_up_real_2 = R_up_1 / pix_ratio
            dy = yy - y
            dy_real = dy / pix_ratio
            volume_up = volume_up + 1 / 3 * math.pi * dy_real * (R_up_real_1**2 + R_up_real_2**2 + R_up_real_1 * R_up_real_2)
            area_up = area_up + math.pi * np.hypot(dy_real, abs(xx - x) / pix_ratio) * (R_up_real_1 + R_up_real_2)
            cv2.fillPoly(imgFix1, [np.array([[y, x], [yy, xx], [yy, xmid], [y, xmid]])], (0, 0, 255))
            cv2.fillPoly(img1, [np.array(
                [[y + x0_cut, x + y0_cut], [yy + x0_cut, xx + y0_cut], [yy + x0_cut, xmid + y0_cut],
                 [y + x0_cut, xmid + y0_cut]])], (0, 0, 255))

        for i in range(len(ltXY_down_2) - 1):
            down = ltXY_down_2[i]
            downdown = ltXY_down_2[i + 1]
            x = down[1]
            y = down[0]
            xx = downdown[1]
            yy = downdown[0]
            R_down_1 = x - xmid
            R_down_2 = xx - xmid
            R_down_real_1 = R_down_1 / pix_ratio
            R_down_real_2 = R_down_2 / pix_ratio
            dy = yy - y
            dy_real = dy / pix_ratio
            volume_down = volume_down + 1 / 3 * math.pi * dy_real * (R_down_real_1**2 + R_down_real_2**2 + R_down_real_1 * R_down_real_2)
            area_down = area_down + math.pi * (R_down_real_1 + R_down_real_2) * np.hypot(dy_real, abs(xx - x) / pix_ratio)
            cv2.fillPoly(imgFix1, [np.array([[y, x], [yy, xx], [yy, xmid], [y, xmid]])], (0, 255, 0))
            cv2.fillPoly(img1, [np.array(
                [[y + x0_cut, x + y0_cut], [yy + x0_cut, xx + y0_cut], [yy + x0_cut, xmid + y0_cut],
                 [y + x0_cut, xmid + y0_cut]])], (0, 255, 0))


    except IndexError:
        return -1, -1, img


    except ValueError:
        return -1, -1, img

    area = (area_up+area_down)/2
    volume = (volume_up+volume_down)/2
    return area, volume, img1

def text_save(full_path, data):
    with open(full_path, 'a') as file:
        for i in range(len(data)):
            s = str(data[i][0]) + "\t" + str(data[i][1]).replace('[', '').replace(']', '')  # delete []
            s = s.replace("'", '').replace(',', '') + '\n'  # delete '，,，add 'enter'
            file.write(s)
        file.write('\n')


class mainWindow:
    def __init__(self, init_window_pointer):
        self.init_window = init_window_pointer
        self.init_window.title("Software for Automated Geometric Analysis of a Generating Droplet")
        self.init_window.geometry('1040x560')
        self.button_readVideo = Button(self.init_window, text="Read video", command=self.read_video)
        self.button_readVideo.place(x=20, y=20, width=120, height=30)
        self.button_analVideo = Button(self.init_window, text="Analyze video", command=self.analyze_video)
        self.button_analVideo.place(x=160, y=20, width=140, height=30)
        self.text1 = Label(self.init_window, text="Pixels/Length(mm)")
        self.text1.place(x=320, y=20, width=190, height=30)
        self.box_pix_ratio = Entry(self.init_window, exportselection = 0)
        self.box_pix_ratio.place(x=510, y=20, width=80, height=30)
        self.box_pix_ratio.insert(0,"376")
        self.text2 = Label(self.init_window, text="Threshold")
        self.text2.place(x=610, y=20, width=100, height=30)
        self.box_threshold_num = Entry(self.init_window, exportselection=0)
        self.box_threshold_num.place(x=710, y=20, width=71, height=30)
        self.box_threshold_num.insert(0, "30")
        self.button_showResultRadius = Button(self.init_window, text="Show results of radius", command=self.showResult_radius)
        self.button_showResultRadius.place(x=50, y=510, width=220, height=40)
        self.button_showResultVolume = Button(self.init_window, text="Show results of volume and surface area", command=self.showResult_volume_area)
        self.button_showResultVolume.place(x=280, y=510, width=365, height=40)
        self.button_saveResult = Button(self.init_window, text="Save results", command=self.save_result)
        self.button_saveResult.place(x=655, y=510, width=120, height=40)
        self.progressbar = Canvas(self.init_window, bg='white', relief='raised', bd=1)
        self.progressbar.place(x=20, y=475, width=1010, height=25)
        self.label_left = Canvas(self.init_window, bg='white')
        self.label_left.place(x=20, y=60, width=500, height=403)
        self.label_right = Canvas(self.init_window, bg='white')
        self.label_right.place(x=535, y=60, width=500, height=403)
        self.v_size = None
        self.fps = None
        self.fourcc = None
        self.frame_num = 0
        self.current_progress = 0
        self.imgTK_L = None
        self.imgTK_R = None

    def read_video(self):
        global frames, v_size
        project_path = os.path.dirname(os.path.abspath(__file__))
        video_name = filedialog.askopenfilename(title='Select a video', initialdir=project_path)
        cap = cv2.VideoCapture(video_name)
        if cap.isOpened():
            success = True
            v_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            self.fps = cap.get(cv2.CAP_PROP_FPS)  # fps
            self.fourcc = cap.get(cv2.CAP_PROP_FOURCC)  # fourcc encoding
            self.frame_num = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        else:
            success = False
            messagebox.showerror("Wrong","Read failed!")
            return None
        frame_index = 0
        while success:
            success, frame = cap.read()
            if not success: break
            frames.append(frame)
            frame_index += 1
            self.update_progress(frame_index / self.frame_num)
        cap.release()
        if frame_index > 0:
            self.update_progress(1)
        messagebox.showinfo("Done","Read compeleted！")

    def update_progress(self,progress:float):
        if progress > 1 or progress < 0:
            pass
        else:
            self.progressbar.delete('ALL')
            if progress <= self.current_progress: self.progressbar.create_rectangle(0, 0, 1300, 26, width=0, fill="white")
            self.progressbar.create_rectangle(0, 0, int(1300*progress), 26, width=0, fill="green")
            self.init_window.update()
            self.current_progress = progress

    def showFrame_radius(self, frame):
        canvawidth = int(self.label_left.winfo_reqwidth())
        canvaheight = int(self.label_left.winfo_reqheight())
        sp = frame.shape
        cvheight = sp[0]
        cvwidth = sp[1]
        if float(cvwidth / cvheight) > float(canvawidth / canvaheight):
            imgCV = cv2.resize(frame, (canvawidth, int(canvawidth * cvheight / cvwidth)),
                              interpolation=cv2.INTER_AREA)
        else:
            imgCV = cv2.resize(frame, (int(canvaheight * cvwidth / cvheight), canvaheight),
                              interpolation=cv2.INTER_AREA)
        imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(imgCV2)
        self.imgTK_L = ImageTk.PhotoImage(image=current_image)
        self.label_left.create_image(0, 0, anchor=NW, image=self.imgTK_L)
        self.init_window.update()

    def showFrame_area(self, frame):
        canvawidth = int(self.label_right.winfo_reqwidth())
        canvaheight = int(self.label_right.winfo_reqheight())
        sp = frame.shape
        cvheight = sp[0]
        cvwidth = sp[1]
        if float(cvwidth / cvheight) > float(canvawidth / canvaheight):
            imgCV = cv2.resize(frame, (canvawidth, int(canvawidth * cvheight / cvwidth)),
                              interpolation=cv2.INTER_AREA)
        else:
            imgCV = cv2.resize(frame, (int(canvaheight * cvwidth / cvheight), canvaheight),
                              interpolation=cv2.INTER_AREA)
        imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(imgCV2)
        self.imgTK_R = ImageTk.PhotoImage(image=current_image)
        self.label_right.create_image(0, 0, anchor=NW, image=self.imgTK_R)
        self.init_window.update()

    def analyze_video(self):
        global frames, fitting_result_area, fitting_result_radius, fitting_result_volume, pix_ratio, threshold_num

        try:
            pix_ratio = int(self.box_pix_ratio.get())
            threshold_num = int(self.box_threshold_num.get())
        except ValueError:
            messagebox.showerror("Wrong", 'Invalid data format！')
            return None

        if len(frames) < 1:
            messagebox.showerror("Wrong", 'No video readed！')
            return None

        window_bgSelect = Toplevel(master=self.init_window)
        bgSelectWindow(window_bgSelect)
        window_bgSelect.mainloop()

        window_ROISelect = Toplevel(master=self.init_window)
        ROISelectWindow(window_ROISelect)
        window_ROISelect.mainloop()

        frame_index = 0
        for frame in frames:
            frame_index += 1
            radius_result, radius_img = analyze_radius(frame.copy())
            fitting_result_radius.append([frame_index, radius_result, radius_img])
            self.showFrame_radius(radius_img)

            area_result, volume_result, volume_img = analyze_area_volume(frame.copy())
            fitting_result_area.append([frame_index, area_result, volume_img])
            fitting_result_volume.append([frame_index, volume_result, volume_img])
            self.showFrame_area(volume_img)

            self.update_progress(frame_index / self.frame_num)

        messagebox.showinfo("Done", "Analysis compeleted！")

    @staticmethod
    def showResult_radius():
        global fitting_result_radius
        xy = [(res[0], res[1]) for res in fitting_result_radius]
        xy_filter = filter(lambda x: x[1] != -1, xy)
        xy_filter_1 = [item for item in xy_filter]
        x = np.array([res[0] for res in xy_filter_1])
        y = np.array([res[1] for res in xy_filter_1])
        pyplot.title("Radius Fitting Result")
        pyplot.xlabel("Time / frame")
        pyplot.ylabel("Radius / mm")
        pyplot.plot(x, y)
        pyplot.show()

    @staticmethod
    def showResult_volume_area():
        global fitting_result_volume, fitting_result_area
        xy1 = [(res[0], res[1]) for res in fitting_result_area]
        xy1_filter = filter(lambda x: x[1] != -1, xy1)
        xy1_filter_1 = [item for item in xy1_filter]
        x1 = np.array([res[0] for res in xy1_filter_1])
        y1 = np.array([res[1] for res in xy1_filter_1])

        xy2 = [(res[0], res[1]) for res in fitting_result_volume]
        xy2_filter = filter(lambda x: x[1] != -1, xy2)
        xy2_filter_1 = [item for item in xy2_filter]
        x2 = np.array([res[0] for res in xy2_filter_1])
        y2 = np.array([res[1] for res in xy2_filter_1])

        pyplot.subplot(2, 1, 1)
        pyplot.plot(x1, y1)
        pyplot.title("Area Fitting Result")
        pyplot.subplot(2, 1, 2)
        pyplot.plot(x2, y2)
        pyplot.title("Volume Fitting Result")
        pyplot.subplots_adjust(hspace=0.35)
        pyplot.show()

    def save_result(self):
        global project_path
        project_path = os.path.dirname(os.path.abspath(__file__))
        file_out = filedialog.asksaveasfilename(title='Save results', initialdir=project_path)

        videoWriterRadius = cv2.VideoWriter(file_out + "_radius.avi", int(cv2.VideoWriter_fourcc('M','P','4','2')), self.fps, v_size)
        videoWriterVolume = cv2.VideoWriter(file_out + "_volume.avi", int(cv2.VideoWriter_fourcc('M','P','4','2')), self.fps, v_size)

        for res in fitting_result_radius: videoWriterRadius.write(res[2])
        for res in fitting_result_volume: videoWriterVolume.write(res[2])

        text_save(file_out + "_radius.txt", fitting_result_radius)
        text_save(file_out + "_volume.txt", fitting_result_volume)
        text_save(file_out + "_area.txt", fitting_result_area)

class bgSelectWindow:
    def __init__(self,init_window_pointer):
        self.init_window = init_window_pointer
        self.init_window.title("Please select a background image……")
        self.init_window.geometry(str(v_size[0]+40)+'x'+str(v_size[1]+120))
        self.image = Canvas(self.init_window, bg="white")
        self.image.place(x=20, y=50, width=v_size[0]+1, height=v_size[1]+1)
        self.button_prev = Button(self.init_window, text="←", command=self.prevFrame)
        self.button_prev.place(x=20, y=10, width=31, height=28)
        self.button_next = Button(self.init_window, text="→", command=self.nextFrame)
        self.button_next.place(x=v_size[0]-10, y=10, width=31, height=28)
        self.button_confirm = Button(self.init_window, text="Select", command=self.bgSelect)
        self.button_confirm.place(x=int(v_size[0]/2-25), y=int(v_size[1]+65), width=110, height=28)
        self.button_import = Button(self.init_window, text="Select from local……", command=self.import_bg)
        self.button_import.place(x=int(v_size[0]/2-55), y=10, width=190, height=28)
        self.current_frame_index = 0
        self.current_image = None
        self.imgTK = None
        self.showFrame(frames[self.current_frame_index])

    def prevFrame(self):
        if self.current_frame_index == 0:
            pass
        else:
            self.current_frame_index -= 1
            self.showFrame(frames[self.current_frame_index])

    def nextFrame(self):
        if self.current_frame_index == len(frames) - 1:
            pass
        else:
            self.current_frame_index += 1
            self.showFrame(frames[self.current_frame_index])

    def showFrame(self, frame):
        self.current_image = frame
        imgCV = frame
        imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)
        imgPIL = Image.fromarray(imgCV2)
        self.imgTK = ImageTk.PhotoImage(image=imgPIL)
        self.image.create_image(0, 0, anchor=NW, image=self.imgTK)

    def bgSelect(self):
        global background
        if self.current_image is not None:
            background = self.current_image
            self.init_window.destroy()
            self.init_window.quit()
            del self.init_window

    def import_bg(self):
        image_name = filedialog.askopenfilename(title=u'Select a background image', initialdir=project_path)
        image_bg = cv2.imread(image_name)
        self.showFrame(image_bg)

class ROISelectWindow:
    def __init__(self, init_window_pointer):
        self.init_window = init_window_pointer
        self.init_window.title("Select a ROI……")
        self.init_window.geometry(str(v_size[0]+40)+'x'+str(v_size[1]+120))
        self.image = Canvas(self.init_window, bg="white")
        self.image.place(x=20, y=50, width=v_size[0]+1, height=v_size[1]+1)
        self.label1 = Label(self.init_window, text="horizontal")
        self.label1.place(x=30, y=10, width=101, height=31)
        self.label2 = Label(self.init_window, text="vertical")
        self.label2.place(x=v_size[0]-240, y=10, width=101, height=31)
        self.box_x1 = Entry(self.init_window, exportselection=0)
        self.box_x1.place(x=140, y=10, width=71, height=31)
        self.box_x2 = Entry(self.init_window, exportselection=0)
        self.box_x2.place(x=220, y=10, width=71, height=31)
        self.box_y1 = Entry(self.init_window, exportselection=0)
        self.box_y1.place(x=v_size[0]-150, y=10, width=71, height=31)
        self.box_y2 = Entry(self.init_window, exportselection=0)
        self.box_y2.place(x=v_size[0]-70, y=10, width=71, height=31)
        self.button_confirm = Button(self.init_window, text="Select", command=self.confirm)
        self.button_confirm.place(x=int(v_size[0]/2+25), y=v_size[1]+70, width=93, height=28)
        self.button_preview = Button(self.init_window, text="Review", command=self.preview)
        self.button_preview.place(x=int(v_size[0]/2-75), y=v_size[1]+70, width=93, height=28)

        self.current_status = 1
        self.point1 = None
        self.point2 = None
        self.ROIrect = None
        self.image.bind("<Button 1>", self.get_location)

        self.imgTK = None
        self.showFrame(background)

    def get_location(self, event):
        global ROI
        x, y = event.x, event.y
        if self.current_status == 1:
            if not self.point1 is None: self.image.delete('point1')
            self.point1 = self.image.create_oval(x - 1, y - 1, x + 1, y + 1, fill="red", tag="point1")
            self.current_status = 2
            self.box_x1.delete(0, 'end')
            self.box_x1.insert(0, str(x))
            self.box_y1.delete(0, 'end')
            self.box_y1.insert(0, str(y))

        else:
            if not self.point2 is None: self.image.delete('point2')
            self.point2 = self.image.create_oval(x - 1, y - 1, x + 1, y + 1, fill="red", tag="point2")
            self.current_status = 1
            self.box_x2.delete(0, 'end')
            self.box_x2.insert(0, str(x))
            self.box_y2.delete(0, 'end')
            self.box_y2.insert(0, str(y))

    def preview(self):
        if not self.ROIrect is None: self.image.delete('ROIrect')
        try:
            x1 = int(self.box_x1.get())
            y1 = int(self.box_y1.get())
            x2 = int(self.box_x2.get())
            y2 = int(self.box_y2.get())
        except ValueError:
            pass
        self.ROIrect = self.image.create_rectangle(x1, y1, x2, y2, fill='', outline='red', tag='ROIrect')

    def confirm(self):
        global ROI
        try:
            ROI[0] = int(self.box_x1.get())
            ROI[1] = int(self.box_y1.get())
            ROI[2] = int(self.box_x2.get())
            ROI[3] = int(self.box_y2.get())
        except ValueError:
            messagebox.showerror("Wrong",'Invalid data format！')
            return None
        self.init_window.destroy()
        self.init_window.quit()
        del self.init_window

    def showFrame(self, frame):
        imgCV = frame
        imgCV2 = cv2.cvtColor(imgCV, cv2.COLOR_BGR2RGBA)
        current_image = Image.fromarray(imgCV2)
        self.imgTK = ImageTk.PhotoImage(image=current_image)
        self.image.create_image(0, 0, anchor=NW, image=self.imgTK)

def gui_start():
    init_window = Tk()
    ZMJ_PORTAL = mainWindow(init_window)

    init_window.mainloop()

if __name__ == '__main__':
    gui_start()
