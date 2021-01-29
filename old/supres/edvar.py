import paddlehub as hub

model = hub.Module(name='edvr')
model.predict('/home/aistudio/test.mp4')
