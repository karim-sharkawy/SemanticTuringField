### TypeError in `renderer.py`
in draw_particles
    pygame.draw.circle(
TypeError: center argument must be a pair of numbers

according to pygame's [draw_circle() documentation](https://www.pygame.org/docs/ref/draw.html#pygame.draw.circle), the `center` argument accepts:

```
center (tuple(int or float, int or float) or list(int or float, int or float) or Vector2(int or float, int or float)) -- center point of the circle as a sequence of 2 ints/floats, e.g. (x, y)
```

center (tuple(int or float, int or float) or list(int or float, int or float) or Vector2(int or float, int or float)) -- center point of the circle as a sequence of 2 ints/floats, e.g. (x, y)
