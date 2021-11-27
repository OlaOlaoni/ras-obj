import streamlit as st
import cv2
from PIL import Image
import numpy as np
import os

from const import CLASSES, COLORS
from settings import DEFAULT_CONFIDENCE_THRESHOLD, DEMO_IMAGE, MODEL, PROTOTXT