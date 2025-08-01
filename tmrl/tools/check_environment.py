# third-party imports
import gymnasium
import cv2
from rtgym.envs.real_time_env import DEFAULT_CONFIG_DICT

# local imports
from tmrl.custom.tm.tm_gym_interfaces import TM2020Interface, TM2020InterfaceLidar
from tmrl.custom.tm.utils.window import WindowInterface
from tmrl.custom.tm.utils.tools import Lidar
import tmrl.config.config_constants as cfg
import logging


def check_env_tm20lidar():
    window_interface = WindowInterface("Trackmania")
    if cfg.SYSTEM != "Windows":
        window_interface.move_and_resize()  # needed on Linux
    lidar = Lidar(window_interface.screenshot())
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020InterfaceLidar
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"img_hist_len": 1, "gamepad": False}
    env = gymnasium.make(cfg.RTGYM_VERSION, config=env_config)
    o, i = env.reset()
    while True:
        o, r, d, t, i = env.step(None)
        #logging.info(f"r:{r}, d:{d}, t:{t}")
        if d or t:
            o, i = env.reset()
        img = window_interface.screenshot()[:, :, :3]
        lidar.lidar_20(img, True)


def show_imgs(imgs, scale=cfg.IMG_SCALE_CHECK_ENV):
    imshape = imgs.shape
    if len(imshape) == 3:  # grayscale
        nb, h, w = imshape
        concat = imgs.reshape((nb*h, w))
        width = int(concat.shape[1] * scale)
        height = int(concat.shape[0] * scale)
        cv2.imshow("Environment", cv2.resize(concat, (width, height), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)
    elif len(imshape) == 4:  # color
        nb, h, w, c = imshape
        concat = imgs.reshape((nb*h, w, c))
        width = int(concat.shape[1] * scale)
        height = int(concat.shape[0] * scale)
        cv2.imshow("Environment", cv2.resize(concat, (width, height), interpolation=cv2.INTER_NEAREST))
        cv2.waitKey(1)


def check_env_tm20full():
    env_config = DEFAULT_CONFIG_DICT.copy()
    env_config["interface"] = TM2020Interface
    env_config["wait_on_done"] = True
    env_config["interface_kwargs"] = {"gamepad": False,
                                      "grayscale": cfg.GRAYSCALE,
                                      "resize_to": (cfg.IMG_WIDTH, cfg.IMG_HEIGHT)}
    env = gymnasium.make(cfg.RTGYM_VERSION, config=env_config)
    o, i = env.reset()
    show_imgs(o[3])
    logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
    while True:
        o, r, d, t, i = env.step(None)
        show_imgs(o[3])
        logging.info(f"r:{r:.2f}, d:{d}, t:{t}, o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")
        if d or t:
            o, i = env.reset()
            show_imgs(o[3])
            logging.info(f"o:[{o[0].item():05.01f}, {o[1].item():03.01f}, {o[2].item():07.01f}, imgs({len(o[3])})]")


if __name__ == "__main__":
    # check_env_tm20lidar()
    check_env_tm20full()
