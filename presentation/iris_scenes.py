"""Iris: two genuine Manim Slides scenes. No measurement text is baked in."""
from manim import (
    DOWN, LEFT, RIGHT, UP, Arrow, Circle, Create, Dot, FadeIn, LaggedStart,
    Line, ManimColor, Rectangle, RoundedRectangle, Text, VGroup, Write,
)
from manim_slides import Slide
import numpy as np

INK, FG, MINT = '#090d11', '#e2ecef', '#9ce8d3'
MUTED, PANEL, BORDER = '#96adb7', '#14212a', '#30444e'
FONT = 'Consolas'


def label(text, size=24, color=FG):
    return Text(text, font=FONT, font_size=size, color=color, line_spacing=1.15)


def card(title, detail='', width=1.85, height=1.22):
    box = RoundedRectangle(width=width, height=height, corner_radius=.10,
                           stroke_color=BORDER, stroke_width=1.5, fill_color=PANEL, fill_opacity=1)
    text = label(title, 23, MINT).move_to(box.get_center() + UP * (.18 if detail else 0))
    if text.width > width - .15:
        text.scale_to_fit_width(width - .15)
    items = [box, text]
    if detail:
        secondary = label(detail, 18, MUTED).move_to(box.get_center() + DOWN * .25)
        if secondary.width > width - .16:
            secondary.scale_to_fit_width(width - .16)
        items.append(secondary)
    return VGroup(*items)


class IrisPipeline(Slide):
    def construct(self):
        self.camera.background_color = ManimColor(INK)
        title = label('A command is only the beginning.', 38).to_edge(UP, buff=.55)
        subtitle = label('Local recognition. Independent guard. Explicit action source.', 22, MUTED).next_to(title, DOWN, buff=.20)
        self.add(title, subtitle)
        stages = [card('Hey Iris', 'native wake'), card('Capture', 'PCM + VAD'),
                  card('VCM', 'whole command'), card('Guard', 'owner + slots'),
                  card('Action', 'one dispatcher'), card('Receipt', 'source + result')]
        nodes = VGroup(*stages).arrange(RIGHT, buff=.30).move_to(UP * .40)
        arrows = VGroup(*[Arrow(stages[i].get_right(), stages[i+1].get_left(), buff=.04,
                                      color=BORDER, stroke_width=2, max_tip_length_to_length_ratio=.22)
                          for i in range(5)])
        self.add(nodes.copy().set_opacity(.20), arrows)
        waveform = VGroup(*[Line([x, -2.1-abs(np.sin(x*7))*.25, 0], [x, -2.1+abs(np.sin(x*7))*.25, 0], color=MINT, stroke_width=2)
                           for x in np.linspace(-6.2, -4.55, 30)])
        note = label('One wake owns one subsequent command.', 24, MUTED).move_to(DOWN*2.85)
        self.play(Create(waveform), FadeIn(stages[0]), run_time=.8)
        self.add_subcaption('The native Hey Iris detector opens one command window.', duration=2)
        self.play(Write(note), run_time=.6)
        self.wait(1)
        self.next_slide()
        self.play(FadeIn(stages[1]), run_time=.8)
        self.wait(.8)
        self.next_slide()
        self.play(FadeIn(stages[2]), run_time=.8)
        self.wait(.8)
        self.next_slide()
        owner = card('Owner / overlap', 'separate companion', width=3.2, height=.95).move_to(DOWN*1.35 + RIGHT*.1)
        branch = Arrow(stages[1].get_bottom(), owner.get_left(), color=BORDER, buff=.08)
        join = Arrow(owner.get_right(), stages[3].get_bottom(), color=MINT, buff=.08)
        self.play(FadeIn(owner), Create(branch), Create(join), FadeIn(stages[3]), run_time=1)
        self.wait(1)
        self.next_slide()
        self.play(FadeIn(stages[4]), FadeIn(stages[5]), run_time=.8)
        source = label('Provider acknowledged ≠ observed device state', 24, MINT).move_to(DOWN*2.13 + RIGHT*1.0)
        self.play(FadeIn(source), run_time=.7)
        self.wait(1.5)
        self.next_slide()


class IrisModel(Slide):
    def construct(self):
        self.camera.background_color = ManimColor(INK)
        title = label('One head. Complete commands.', 38).to_edge(UP, buff=.55)
        subtitle = label('A finite acoustic classifier — not a transcription system.', 22, MUTED).next_to(title, DOWN, buff=.20)
        self.add(title, subtitle)
        # A literal time × mel-bin lattice, not an invented measured spectrogram.
        grid = VGroup(*[Rectangle(width=.12, height=.17, stroke_width=0,
                                  fill_color=MINT, fill_opacity=.18 + .65*abs(np.sin(t*.55 + b*.41)))
                        .move_to([-5.5 + t*.15, -.50+b*.20, 0])
                        for t in range(14) for b in range(9)])
        axis = label('40 mel bins × time', 20, MUTED).next_to(grid, DOWN, buff=.22)
        self.play(LaggedStart(*[FadeIn(row) for row in grid], lag_ratio=.003), FadeIn(axis), run_time=.9)
        self.wait(.8)
        self.next_slide()
        stem = card('Stride 2', 'causal stem k5', width=2, height=1.35).move_to(LEFT*1.4 + UP*.28)
        flow1 = Arrow(grid.get_right(), stem.get_left(), color=BORDER, buff=.16)
        self.play(FadeIn(stem), Create(flow1), run_time=.8)
        self.wait(.8)
        self.next_slide()
        blocks = VGroup(*[card(f'd = {d}', '2 × DW/PW', width=1.65, height=.80) for d in (1,2,4,8)]).arrange(DOWN, buff=.08).move_to(RIGHT*1.28 + UP*.48)
        block_title = label('Width 160 · k7', 20, MUTED).move_to(LEFT*1.4 + DOWN*1.1)
        flow2 = Arrow(stem.get_right(), blocks.get_left(), color=BORDER, buff=.16)
        self.play(Create(flow2), LaggedStart(*[FadeIn(b) for b in blocks], lag_ratio=.2), FadeIn(block_title), run_time=1.3)
        self.wait(1)
        self.next_slide()
        pooling = card('Mean + max', 'valid lengths', width=2, height=1.35).move_to(RIGHT*4.25 + UP*.28)
        flow3 = Arrow(blocks.get_right(), pooling.get_left(), color=BORDER, buff=.16)
        self.play(FadeIn(pooling), Create(flow3), run_time=.8)
        self.wait(.8)
        self.next_slide()
        output = VGroup(label('320 pooled features', 23, MUTED), label('32-way softmax', 28, MINT),
                        label('31 commands + UNKNOWN', 23), label('No separately predicted slots.', 22, MUTED)).arrange(DOWN, buff=.24).move_to(DOWN*2.60 + LEFT*.3)
        flow4 = Arrow(pooling.get_bottom(), output.get_top() + RIGHT*2, color=BORDER, buff=.16)
        self.play(FadeIn(output), Create(flow4), run_time=1)
        self.wait(1.5)
        self.next_slide()
