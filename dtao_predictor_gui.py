#!/usr/bin/env python3
# GUI
import argparse
import tkinter as tk
from src.gui import DTAOPredictorGUI

def parse_args():
    parser = argparse.ArgumentParser(description="dTAO Price Predictor GUI")
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock backend data instead of real Bittensor/Taostats data",
    )
    return parser.parse_args()

def main():
    """GUI"""
    args = parse_args()
    root = tk.Tk()
    root.title("dTAO Price Predictor")
    app = DTAOPredictorGUI(root, use_mock=args.mock)
    root.mainloop()

if __name__ == "__main__":
    main()
