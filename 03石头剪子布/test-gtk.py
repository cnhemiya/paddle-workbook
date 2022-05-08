#!/usr/bin/python3
"""
LICENSE: MulanPSL2
AUTHOR:  cnhemiya@qq.com
DATE:    2022-04-19 15:50
文档说明: 测试, GTK 界面
"""


import os
import random
import paddle
import mod.dataset
import mod.config
import mod.utils
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk


test_ui_str = """
<?xml version="1.0" encoding="UTF-8"?>
<!-- Generated with glade 3.38.2 -->
<interface>
  <requires lib="gtk+" version="3.24"/>
  <object class="GtkWindow" id="main_wnd">
    <property name="can-focus">False</property>
    <property name="title" translatable="yes">测试图像</property>
    <property name="default-width">800</property>
    <property name="default-height">600</property>
    <child>
      <object class="GtkBox">
        <property name="visible">True</property>
        <property name="can-focus">False</property>
        <property name="orientation">vertical</property>
        <child>
          <object class="GtkBox">
            <property name="visible">True</property>
            <property name="can-focus">False</property>
            <child>
              <object class="GtkButton" id="choose_btn">
                <property name="label" translatable="yes">选择模型</property>
                <property name="visible">True</property>
                <property name="can-focus">True</property>
                <property name="receives-default">True</property>
              </object>
              <packing>
                <property name="expand">False</property>
                <property name="fill">True</property>
                <property name="position">0</property>
              </packing>
            </child>
            <child>
              <object class="GtkLabel" id="model_path_lbe">
                <property name="visible">True</property>
                <property name="can-focus">False</property>
                <property name="label" translatable="yes">无模型</property>
              </object>
              <packing>
                <property name="expand">True</property>
                <property name="fill">True</property>
                <property name="position">1</property>
              </packing>
            </child>
          </object>
          <packing>
            <property name="expand">False</property>
            <property name="fill">True</property>
            <property name="position">0</property>
          </packing>
        </child>
        <child>
          <object class="GtkBox">
            <property name="visible">True</property>
            <property name="can-focus">False</property>
            <child>
              <object class="GtkButton" id="test_btn">
                <property name="label" translatable="yes">随机测试</property>
                <property name="visible">True</property>
                <property name="can-focus">True</property>
                <property name="receives-default">True</property>
              </object>
              <packing>
                <property name="expand">False</property>
                <property name="fill">True</property>
                <property name="position">0</property>
              </packing>
            </child>
            <child>
              <object class="GtkLabel" id="result_lbe">
                <property name="visible">True</property>
                <property name="can-focus">False</property>
                <property name="label" translatable="yes">测试结果</property>
              </object>
              <packing>
                <property name="expand">True</property>
                <property name="fill">True</property>
                <property name="position">1</property>
              </packing>
            </child>
          </object>
          <packing>
            <property name="expand">False</property>
            <property name="fill">True</property>
            <property name="position">1</property>
          </packing>
        </child>
        <child>
          <object class="GtkImage" id="test_img">
            <property name="visible">True</property>
            <property name="can-focus">False</property>
            <property name="stock">gtk-missing-image</property>
            <property name="icon_size">6</property>
          </object>
          <packing>
            <property name="expand">True</property>
            <property name="fill">True</property>
            <property name="position">2</property>
          </packing>
        </child>
      </object>
    </child>
  </object>
</interface>
"""


class TestWindow(Gtk.Window):
    def __init__(self):
        Gtk.Window.__init__(self, title="测试图像")

        self.builder = Gtk.Builder.new_from_string(test_ui_str, -1)
        self.main_wnd = self.builder.get_object("main_wnd")
        self.choose_btn = self.builder.get_object("choose_btn")
        self.model_path_lbe = self.builder.get_object("model_path_lbe")
        self.test_btn = self.builder.get_object("test_btn")
        self.result_lbe = self.builder.get_object("result_lbe")
        self.test_img = self.builder.get_object("test_img")

        self.main_wnd.connect("destroy", Gtk.main_quit)
        self.choose_btn.connect("clicked", self.on_choose_btn_clicked)
        self.test_btn.connect("clicked", self.on_test_btn_clicked)

        self.model_filename = ""
        self.net = self.init_net()
        self.test_image_paths, self.test_labels = mod.dataset.ImageClass.parse_dataset(
            mod.config.DATASET_PATH, mod.config.TEST_DATA_PATH, True)

    def run(self):
        self.main_wnd.show_all()
        Gtk.main()

    def init_net(self):
        """
        初始化模型
        """
        mod.config.user_cude(True)
        net = mod.config.net()
        return net

    def test_net(self):
        """
        模型测试
        """
        self.net.eval()
        idx = random.randint(0, len(self.test_image_paths)-1)
        self.test_img.set_from_file(self.test_image_paths[idx])
        image, label = mod.dataset.ImageClass.get_item(
            self.test_image_paths[idx], self.test_labels[idx], mod.config.transform())
        predict_result = self.net(mod.config.image_to_tensor(image))
        class_id = mod.utils.predict_to_class(predict_result)
        result_txt = mod.config.CLASS_TXT[class_id]
        label_txt = mod.config.CLASS_TXT[label]
        self.result_lbe.set_text("预测分类:  {}    {},        实际分类:  {}    {}".format(
            class_id, result_txt, label, label_txt))

    def choose_file(self, filename: str):
        if str == "":
            self.model_path_lbe = ""
            self.model_filename = ""
        else:
            base_name = os.path.basename(filename)
            self.model_path_lbe.set_text(base_name)
            self.model_filename = filename
            print("读取模型参数。。。")
            self.net.set_state_dict(paddle.load(filename))
            print("模型参数读取完毕！")

    def on_choose_btn_clicked(self, widget):
        dialog = Gtk.FileChooserDialog("请选择模型文件", self,
                                       Gtk.FileChooserAction.OPEN,
                                       (Gtk.STOCK_CANCEL, Gtk.ResponseType.CANCEL,
                                        Gtk.STOCK_OPEN, Gtk.ResponseType.OK))

        self.add_filters(dialog)

        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            self.choose_file(dialog.get_filename())
        elif response == Gtk.ResponseType.CANCEL:
            self.choose_file("")

        dialog.destroy()

    def add_filters(self, dialog):
        filter_pdparams = Gtk.FileFilter()
        filter_pdparams.set_name("pdparams files")
        filter_pdparams.add_pattern("*.pdparams")
        dialog.add_filter(filter_pdparams)

        filter_any = Gtk.FileFilter()
        filter_any.set_name("Any files")
        filter_any.add_pattern("*")
        dialog.add_filter(filter_any)

    def on_test_btn_clicked(self, widget):
        self.test_net()


def main():
    window = TestWindow()
    window.run()


if __name__ == "__main__":
    main()
