import wx
import wx.xrc
from skimage import io, transform
import tensorflow as tf
import numpy as np
import cv2


class Module:
    def __init__(self):
        self.fruit_dict = {0: '苹果', 1: '香蕉', 2: '橙子'}
        self.w = 100
        self.h = 100
        self.c = 3

    # 读取图片
    def read_image(self, path):
        img = io.imread(path)
        img = transform.resize(img, (self.w, self.h))
        return np.asarray(img)

    # 识别图片
    def classification(self, path):
        with tf.Session() as sess:
            data = list()
            data1 = self.read_image(path)
            data.append(data1)
            saver = tf.train.import_meta_graph('./module//model.meta')
            saver.restore(sess, tf.train.latest_checkpoint('./module/'))
            graph = tf.get_default_graph()
            x = graph.get_tensor_by_name("x:0")
            feed_dict = {x: data}
            logits = graph.get_tensor_by_name("logits_tag:0")
            classification_result = sess.run(logits, feed_dict)
            # 预测矩阵
            print(classification_result)
            # 预测矩阵每一行最大值的索引
            print(tf.argmax(classification_result, 1).eval())
            # 得到索引对应水果的分类
            output = tf.argmax(classification_result, 1).eval()
            result = "预测结果: \n" + self.fruit_dict[output[0]]
        return result

    # 利用opencv调用摄像头
    @staticmethod
    def select_files():
        cap = cv2.VideoCapture(0)
        while True:
            ret, frame = cap.read()
            cv2.imshow("capture", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                cv2.imwrite("capture.jpg", frame)
                break
        cap.release()
        cv2.destroyAllWindows()


class MyFrame1 (wx.Frame):
    def __init__(self, parent):
        wx.Frame.__init__(self, parent, id=wx.ID_ANY, title=u"水果识别", pos=wx.DefaultPosition, size=wx.Size(720,480), style=wx.DEFAULT_FRAME_STYLE | wx.TAB_TRAVERSAL)
        self.SetSizeHints(wx.DefaultSize, wx.DefaultSize)
        self.icon = wx.Icon('icon.ico', type=wx.BITMAP_TYPE_ICO)
        self.SetIcon(self.icon)
        bSizer1 = wx.BoxSizer(wx.VERTICAL)
        img_big = wx.Image("./init.jpg", wx.BITMAP_TYPE_ANY).Scale(720, 360).ConvertToBitmap()
        self.m_bitmap1 = wx.StaticBitmap(self, wx.ID_ANY, img_big, wx.DefaultPosition, wx.DefaultSize, 0)
        bSizer1.Add(self.m_bitmap1, 1, wx.ALIGN_CENTER_HORIZONTAL|wx.EXPAND|wx.BOTTOM|wx.RIGHT|wx.LEFT, 5)
        gSizer1 = wx.GridSizer(0, 2, 0, 0)
        self.m_staticText1 = wx.StaticText(self, wx.ID_ANY, u"从文件获取", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText1.Wrap(-1)
        gSizer1.Add(self.m_staticText1, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_BOTTOM, 5)
        self.m_staticText2 = wx.StaticText(self, wx.ID_ANY, u"从相机获取（按'q'进行捕捉）", wx.DefaultPosition, wx.DefaultSize, 0)
        self.m_staticText2.Wrap(-1)
        gSizer1.Add(self.m_staticText2, 0, wx.ALL|wx.ALIGN_CENTER_HORIZONTAL|wx.ALIGN_BOTTOM, 5 )
        self.m_filePicker1 = wx.FilePickerCtrl(self, wx.ID_ANY, wx.EmptyString, u"选择jpg文件", u"*.jpg", wx.DefaultPosition, wx.DefaultSize, wx.FLP_DEFAULT_STYLE|wx.FLP_OPEN)
        gSizer1.Add(self.m_filePicker1, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)
        self.m_button1 = wx.Button(self, wx.ID_ANY, u"相机", wx.DefaultPosition, wx.DefaultSize, 0)
        gSizer1.Add(self.m_button1, 0, wx.ALIGN_CENTER_HORIZONTAL, 5)
        bSizer1.Add(gSizer1, 1, wx.EXPAND, 5)
        self.SetSizer(bSizer1)
        self.Layout()
        self.Centre(wx.BOTH)
        self.m_filePicker1.Bind(wx.EVT_FILEPICKER_CHANGED, self.m_filePicker1OnFileChanged)
        self.m_button1.Bind(wx.EVT_BUTTON, self.m_button1OnButtonClick)
        self.module = Module()

    def __del__(self):
        pass

    @staticmethod
    def __resizeBitmap(image, width=100, height=100):
        bmp = image.Scale(width, height).ConvertToBitmap()
        return bmp

    def m_filePicker1OnFileChanged(self, event):
        path = self.m_filePicker1.GetPath()
        img_ori = wx.Image(path, wx.BITMAP_TYPE_ANY)
        self.m_bitmap1.SetBitmap(self.__resizeBitmap(img_ori, 250, 250))
        string = self.module.classification(path)
        wx.MessageBox(string, '识别结果', wx.OK | wx.ICON_INFORMATION)

    def m_button1OnButtonClick(self, event):
        self.module.select_files()
        img_ori = wx.Image("./capture.jpg", wx.BITMAP_TYPE_ANY)
        self.m_bitmap1.SetBitmap(self.__resizeBitmap(img_ori, 250, 250))
        string = self.module.classification("./capture.jpg")
        wx.MessageBox(string, '识别结果', wx.OK | wx.ICON_INFORMATION)


if __name__ == '__main__':
    app = wx.App(False)
    frame = MyFrame1(None)
    frame.Show(True)
    app.MainLoop()
