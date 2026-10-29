"""Tests for pure helpers in the regular poster."""
from app.graphs.poster import _safe_image_prompt

def test_safe_image_prompt_empty():
    result = _safe_image_prompt("")
    assert "realistic photo of modern restored kitchen" in result
    assert "no clutter" in result
    assert "no text" in result

def test_safe_image_prompt_strips_before_after():
    raw = "before and after kitchen cabinet restoration, modern style"
    result = _safe_image_prompt(raw)
    assert "before" not in result.lower()
    assert "after" not in result.lower()
    assert "realistic photography" in result
    assert "single photo" in result
    assert "no text" in result

def test_safe_image_prompt_strips_split_and_comparison():
    raw = "split image comparison of old kitchen and new kitchen"
    result = _safe_image_prompt(raw)
    lower = result.lower()
    assert "split" not in lower
    assert "comparison" not in lower
    assert "old" not in lower or "kitchen" not in lower  # may remain partially, but key words gone
    assert "single photo of modern restored kitchen" in result

def test_safe_image_prompt_keeps_good_and_adds_suffix():
    raw = "modern white kitchen cabinets with wood countertop"
    result = _safe_image_prompt(raw)
    assert "modern white kitchen cabinets" in result
    assert "realistic photography" in result
    assert "no clutter" in result
    assert "no text" in result
    assert len(result) < 400  # stays reasonable

def test_safe_image_prompt_truncates_long():
    raw = "a very long realistic photo description of a beautiful modern kitchen with many details about cabinets doors handles lighting and more extra words here to make it long"
    result = _safe_image_prompt(raw)
    assert len(result) < 350
    assert "realistic photography" in result
