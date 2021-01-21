import paddlehub as hub

model = hub.Module(name='edvr')
model.predict('/home/aistudio/plane/test-gs/test.mp4')
