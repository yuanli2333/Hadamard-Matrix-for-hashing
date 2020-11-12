import torch
import torch.utils.data as data
import os
import logging
import nvvl
import pynvvl
from torchvision import transforms
import cupy


class Video_dataset(data.Dataset):

    def __init__(self,
                 video_prefix,  # video root
                 txt_list,      # video list
                 clip_length,
                 device_id,     # gpu id for load video
                 video_transform=None,
                 return_item_subpath=False,
                 transform=None
                 ):
        super(Video_dataset, self).__init__()
        self.video_prefix = video_prefix
        self.clip_length = clip_length
        self.video_transform = video_transform
        self.return_item_subpath = return_item_subpath
        self.device_id = device_id
        self.transform = transforms.Compose([transforms.ToTensor()])  # you can add to the list all the transformations you need.

        self.video_list = self._get_video_list(video_prefix=self.video_prefix, txt_list=txt_list,)   # [vid, label, video_subpath]
        #self.loader = pynvvl.NVVLVideoLoader(device_id=self.device_id)
        #self.loader = pynvvl.NVVLVideoLoader(device_id=self.device_id, log_level='error')
        """
        self.processing = {"input": nvvl.ProcessDesc(
            type = 'half',
            height = 224,
            width = 224,
            random_crop = False,
            random_flip = False,
            color_space = "RGB",
            dimension_order = "cfhw",
            normalized = True
        )}"""

    def __getitem__(self, index):

        try:
            clip_input, label, vid_subpath = self._video_load(index)

            if self.return_item_subpath:
                return clip_input, label, vid_subpath
            else:
                return clip_input, label

        except Exception as e:
            index = index - 1 if index > 0 else index + 1
            print("Video_dataset: %s, skip it"%e)
            return self.__getitem__(index)


    def __len__(self):
        return len(self.video_list)


    def _video_load(self, index):
        # get current video info
        v_id, label, vid_subpath = self.video_list[index]
        video_path = os.path.join(self.video_prefix, vid_subpath)
        #video = nvvl.VideoDataset(video_path,
                                  #sequence_length=self.clip_length,
                                  #device_id=self.device_id,
                                  #processing = self.processing)
        loader = pynvvl.NVVLVideoLoader(device_id=self.device_id)
        if os.path.getsize(video_path)<300:
            print("Null video: {}, skip it".format(video_path))
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        else:
            n_frames = loader.frame_count(video_path)
            if n_frames == 0:
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)
            random_start_frame = int(2)

            video = loader.read_sequence(
                filename=video_path,
                frame=random_start_frame,
                count=self.clip_length,
                scale_height=224,
                scale_width=224,
                crop_height=224,
                crop_width=224,
                horiz_flip=True,
                scale_method='Linear',
                normalized=True
                )
            video = cupy.asnumpy(video)
            video = torch.from_numpy(video)

        return video, label, vid_subpath


    def _get_video_list(self,
                        video_prefix,
                        txt_list,
                        check_video=False,
                        cached_info_path=None):
        # formate:
        # [v_id, label, video_subpath]
        assert os.path.exists(video_prefix), "VideoIter:: failed to locate: `{}'".format(video_prefix)
        assert os.path.exists(txt_list), "VideoIter:: failed to locate: `{}'".format(txt_list)

        # building dataset
        video_list = []    # [v_id, label, video_subpath]
        with open(txt_list) as f:
            lines = f.read().splitlines()
            logging.info("VideoIter:: found {} videos in `{}'".format(len(lines), txt_list))
            for i, line in enumerate(lines):
                v_id, label, video_subpath = line.split()
                video_subpath=video_subpath.replace('.avi', '.mp4')  # format should be '.mp4' for nvvl
                #print(video_subpath)
                video_path = os.path.join(video_prefix, video_subpath)
                if not os.path.exists(video_path):
                    logging.warning("VideoIter:: >> cannot locate `{}'".format(video_path))
                    continue
                info = [int(v_id), int(label), video_subpath]
                video_list.append(info)

        return video_list



if __name__=="__main__":
    # test by UCF101
    data_root = '../dataset/UCF101'
    video_profix = os.path.join(data_root, 'raw', 'nvvl_data')
    train_txt_list = os.path.join(data_root, 'raw', 'list_cvt', 'trainlist01.txt')
    clip_length = 8

    train_dataset = Video_dataset(video_prefix=video_profix,
                                  txt_list=train_txt_list,
                                  clip_length=clip_length,
                                  device_id=0,
                                  return_item_subpath=True,
                                  transform=transforms.ToTensor()
                                  )


    print('number of video: %d' % len(train_dataset))

    train_loader = data.DataLoader(train_dataset, batch_size = 12, shuffle = True)

    print(len(train_loader))

    for data, target, file in train_loader:
        print(target)
        #print(file)
        #print(len(target))
        print(data.shape)



