import cv2
import torch


class CameraDetector:
    def __init__(self, model_path, device='cpu'):   #当创建类的对象时，调动__init__方法，运行这个代码块内的代码；此代码块用于正确加载模型
        """
        初始化相机检测器
        :param model_path: 本地 YOLOv5 模型的路径，例如 'yolov5s.pt'
        :param device: 运行设备，'cpu' 或 'cuda'
        """
        self.device = torch.device(device)      #torch.device函数，通过定义类的实例时提供的字符串识别出运行设备，并以self.device变量表示该设备
        try:    # try子句用于检测包含的代码块中是否有异常
            # 从本地加载 YOLOv5 自定义模型
            self.model = torch.hub.load('.', 'custom', path=model_path, source='local') # 用于从指定路径加载预训练模型
            self.model.to(self.device)  #将预训练模型移动至指定设备上
            self.model.eval()  #评估模式，即可使用模块对未知数据进行推理
            print("YOLOv5 模型加载成功！")
        except Exception as e:    #except子句在try子句发现异常后执行
            print(f"模型加载失败，错误信息：{e}")
            raise

    def detect_from_camera(self, camera_index=0):  #定义另一个函数，当类被创建后可调用；用于从摄像头捕获视频流
        """
        从本地摄像头进行检测
        :param camera_index: 本地摄像头的索引，默认 0
        """
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened(): #  .isOpened成员函数
                print("无法打开本地摄像头，请检查设备！")
                return
        except Exception as e:
            print(f"打开摄像头时出错，错误信息：{e}")
            return

        print("开始从本地摄像头读取...")
        while True:
            try:
                ret, frame = cap.read()
                if not ret:
                    print("无法读取视频帧")
                    break

                # 模型推理
                with torch.no_grad(): #由于处于评估模式，无需更新参数，故此处禁用梯度计算
                    results = self.model(frame) # 使用模型对捕获视频流进行推理，将处理后的数据用results参数接受

                    # 显示带有检测结果的画面
                    img_with_boxes = results.render()[0]  
                    cv2.imshow('YOLOv5 Detection - Camera', img_with_boxes)
            except Exception as e:
                print(f"推理过程中出现错误，错误信息：{e}")

            # 按 'q' 键退出
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print("摄像头检测结束！")


# 主程序入口
if __name__ == "__main__":
    model_path = 'yolov5s.pt'  # 模型路径，请确保该文件存在本地
    detector = CameraDetector(model_path=model_path, device='cpu')
    detector.detect_from_camera()