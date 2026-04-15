# -*- coding:utf-8 -*-
"""
@Time: 2022/03/08 13:01
@Author: KI
@File: main.py
@Motto: Hungry And Humble
"""
from args import args_parser
from server import PerFed
from src.core import AppConfig

def main(args_override=None, cfg: AppConfig = None):
    args = args_parser()
    
    # Override args with provided overrides (from YAML)
    if args_override:
        if isinstance(args_override, dict):
            for k, v in args_override.items():
                setattr(args, k, v)
        else:
            for k, v in vars(args_override).items():
                setattr(args, k, v)
                
    # Override args with cfg if not explicitly set in args_override?
    # Or let PerFed handle it?
    # PerFed currently uses args as base.
    # Let's pass both to PerFed and let it handle priority.
    
    perFed = PerFed(args, cfg=cfg)
    perFed.server()
    perFed.global_test()

class PerFedAvgWrapper:
    """Wrapper for ExtAdapter"""
    def __init__(self, args, cfg=None):
        self.args = args
        self.cfg = cfg
        
    def run(self):
        main(args_override=self.args, cfg=self.cfg)
