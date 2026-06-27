import os
import cv2
import numpy as np
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from config import settings
from core.detector import FaceDetectorYuNet
from core.recognizer import FaceRecognizerSFace
from core.reid import PersonReIDExtractor, YoloPersonDetector

def test_resolve_backend_cpu():
    """Verifies that resolve_dnn_backend_target returns CPU when USE_CUDA is False."""
    orig_use_cuda = settings.USE_CUDA
    settings.USE_CUDA = False
    
    backend, target, desc = settings.resolve_dnn_backend_target()
    assert backend == cv2.dnn.DNN_BACKEND_OPENCV
    assert target == cv2.dnn.DNN_TARGET_CPU
    assert "CPU" in desc
    
    settings.USE_CUDA = orig_use_cuda

@patch("cv2.cuda.getCudaEnabledDeviceCount")
def test_resolve_backend_cuda_fp32(mock_device_count):
    """Verifies that resolve_dnn_backend_target returns CUDA target when CUDA is available and FP16 is False."""
    orig_use_cuda = settings.USE_CUDA
    orig_fp16 = settings.CUDA_FP16
    
    settings.USE_CUDA = True
    settings.CUDA_FP16 = False
    mock_device_count.return_value = 1
    
    # We mock cv2.cuda.setDevice as well to avoid calling it in CPU test runs
    with patch("cv2.cuda.setDevice") as mock_set_device:
        backend, target, desc = settings.resolve_dnn_backend_target()
        assert backend == cv2.dnn.DNN_BACKEND_CUDA
        assert target == cv2.dnn.DNN_TARGET_CUDA
        assert "CUDA" in desc
        mock_set_device.assert_called_once_with(settings.CUDA_DEVICE_INDEX)
        
    settings.USE_CUDA = orig_use_cuda
    settings.CUDA_FP16 = orig_fp16

@patch("cv2.cuda.getCudaEnabledDeviceCount")
def test_resolve_backend_cuda_fp16(mock_device_count):
    """Verifies that resolve_dnn_backend_target returns CUDA FP16 target when FP16 is enabled."""
    orig_use_cuda = settings.USE_CUDA
    orig_fp16 = settings.CUDA_FP16
    
    settings.USE_CUDA = True
    settings.CUDA_FP16 = True
    mock_device_count.return_value = 1
    
    with patch("cv2.cuda.setDevice") as mock_set_device:
        backend, target, desc = settings.resolve_dnn_backend_target()
        assert backend == cv2.dnn.DNN_BACKEND_CUDA
        assert target == cv2.dnn.DNN_TARGET_CUDA_FP16
        assert "FP16" in desc
        mock_set_device.assert_called_once_with(settings.CUDA_DEVICE_INDEX)
        
    settings.USE_CUDA = orig_use_cuda
    settings.CUDA_FP16 = orig_fp16

@patch("cv2.FaceDetectorYN.create")
def test_yunet_cuda_fallback(mock_create):
    """Verifies that YuNet FaceDetector handles CUDA initialization failures and falls back to CPU."""
    # First call to create (with CUDA) fails, second call (CPU fallback) succeeds
    mock_detector_instance = MagicMock()
    mock_create.side_effect = [RuntimeError("CUDA Error"), mock_detector_instance]
    
    # Mock settings.resolve_dnn_backend_target to pretend we chose CUDA
    with patch("config.settings.resolve_dnn_backend_target") as mock_resolve:
        mock_resolve.return_value = (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "CUDA (Test)")
        
        # Instantiate FaceDetectorYuNet
        detector = FaceDetectorYuNet(settings.YUNET_MODEL_PATH)
        detector._init_detector((640, 480))
        
        # Check that FaceDetectorYN.create was called twice (first with CUDA, then with CPU)
        assert mock_create.call_count == 2
        first_call = mock_create.call_args_list[0]
        second_call = mock_create.call_args_list[1]
        
        assert first_call[1]["backend_id"] == cv2.dnn.DNN_BACKEND_CUDA
        assert second_call[1]["backend_id"] == cv2.dnn.DNN_BACKEND_OPENCV
        assert second_call[1]["target_id"] == cv2.dnn.DNN_TARGET_CPU

@patch("cv2.FaceRecognizerSF.create")
def test_sface_cuda_fallback(mock_create):
    """Verifies that SFace FaceRecognizer handles CUDA failures and falls back to CPU."""
    mock_recognizer_instance = MagicMock()
    mock_create.side_effect = [RuntimeError("CUDA Device Error"), mock_recognizer_instance]
    
    with patch("config.settings.resolve_dnn_backend_target") as mock_resolve:
        mock_resolve.return_value = (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "CUDA (Test)")
        
        recognizer = FaceRecognizerSFace(settings.SFACE_MODEL_PATH)
        
        assert mock_create.call_count == 2
        assert mock_create.call_args_list[0][1]["backend_id"] == cv2.dnn.DNN_BACKEND_CUDA
        assert mock_create.call_args_list[1][1]["backend_id"] == cv2.dnn.DNN_BACKEND_OPENCV

@patch("cv2.dnn.readNet")
def test_reid_extractor_cuda_fallback(mock_read_net):
    """Verifies that PersonReIDExtractor falls back to CPU if setting backend/target or dry run fails."""
    mock_net = MagicMock()
    mock_read_net.return_value = mock_net
    
    # Let net.forward raise an error during dry run to simulate a runtime CUDA failure
    mock_net.forward.side_effect = RuntimeError("CUDA Out of Memory during dry run")
    
    with patch("config.settings.resolve_dnn_backend_target") as mock_resolve:
        mock_resolve.return_value = (cv2.dnn.DNN_BACKEND_CUDA, cv2.dnn.DNN_TARGET_CUDA, "CUDA (Test)")
        
        extractor = PersonReIDExtractor(settings.REID_MODEL_PATH)
        
        # net.setPreferableBackend should be called first with CUDA, then with CPU
        mock_net.setPreferableBackend.assert_any_call(cv2.dnn.DNN_BACKEND_CUDA)
        mock_net.setPreferableBackend.assert_any_call(cv2.dnn.DNN_BACKEND_OPENCV)
        mock_net.setPreferableTarget.assert_any_call(cv2.dnn.DNN_TARGET_CPU)
