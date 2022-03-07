from yacs.config import CfgNode as CN

tank_cfg = CN()

tank_cfg.META_ARC = "tank_test_config"

tank_cfg.scenes = ("Family", "Francis", "Horse", "Lighthouse", "M60", "Panther", "Playground", "Train", "Auditorium", "Ballroom", "Courtroom", "Museum", "Palace", "Temple")

tank_cfg.Family = CN()
tank_cfg.Family.max_h = 1080
tank_cfg.Family.max_w = 2048
tank_cfg.Family.conf = [0.4, 0.6, 0.9]

tank_cfg.Francis = CN()
tank_cfg.Francis.max_h = 1080
tank_cfg.Francis.max_w = 2048
tank_cfg.Francis.conf = [0.4, 0.6, 0.95]

tank_cfg.Horse = CN()
tank_cfg.Horse.max_h = 1080
tank_cfg.Horse.max_w = 2048
tank_cfg.Horse.conf = [0.05, 0.1, 0.6]

tank_cfg.Lighthouse = CN()
tank_cfg.Lighthouse.max_h = 1080
tank_cfg.Lighthouse.max_w = 2048
tank_cfg.Lighthouse.conf = [0.5, 0.6, 0.9]

tank_cfg.M60 = CN()
tank_cfg.M60.max_h = 1080
tank_cfg.M60.max_w = 2048
tank_cfg.M60.conf = [0.4, 0.7, 0.9]

tank_cfg.Panther = CN()
tank_cfg.Panther.max_h = 896
tank_cfg.Panther.max_w = 1216
tank_cfg.Panther.conf = [0.1, 0.15, 0.9]

tank_cfg.Playground = CN()
tank_cfg.Playground.max_h = 1080
tank_cfg.Playground.max_w = 2048
tank_cfg.Playground.conf = [0.5, 0.7, 0.9]

tank_cfg.Train = CN()
tank_cfg.Train.max_h = 1080
tank_cfg.Train.max_w = 2048
tank_cfg.Train.conf = [0.3, 0.6, 0.95]

tank_cfg.Auditorium = CN()
tank_cfg.Auditorium.max_h = 1080
tank_cfg.Auditorium.max_w = 2048
tank_cfg.Auditorium.conf = [0.0, 0.0, 0.4]

tank_cfg.Ballroom = CN()
tank_cfg.Ballroom.max_h = 1080
tank_cfg.Ballroom.max_w = 2048
tank_cfg.Ballroom.conf = [0.0, 0.0, 0.5]

tank_cfg.Courtroom = CN()
tank_cfg.Courtroom.max_h = 1080
tank_cfg.Courtroom.max_w = 2048
tank_cfg.Courtroom.conf = [0.0, 0.0, 0.4]

tank_cfg.Museum = CN()
tank_cfg.Museum.max_h = 1080
tank_cfg.Museum.max_w = 2048
tank_cfg.Museum.conf = [0.0, 0.0, 0.7]

tank_cfg.Palace = CN()
tank_cfg.Palace.max_h = 1080
tank_cfg.Palace.max_w = 2048
tank_cfg.Palace.conf = [0.0, 0.0, 0.7]

tank_cfg.Temple = CN()
tank_cfg.Temple.max_h = 1080
tank_cfg.Temple.max_w = 2048
tank_cfg.Temple.conf = [0.0, 0.0, 0.4]