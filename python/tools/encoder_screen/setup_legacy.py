"""Create a separate Unity 2022/Sentis 1.2 compatibility diagnostic project."""
from pathlib import Path
from common import ROOT, WORK, dump


def setup():
    project=WORK/'unity2022'
    assets=project/'Assets'; (assets/'Editor').mkdir(parents=True,exist_ok=True)
    dump(project/'Packages/manifest.json',{'dependencies':{
        'com.unity.sentis':'1.2.0-exp.2','com.unity.nuget.newtonsoft-json':'3.2.1',
        'com.unity.modules.audio':'1.0.0','com.unity.modules.imgui':'1.0.0',
        'com.unity.modules.jsonserialize':'1.0.0','com.unity.modules.physics':'1.0.0'}})
    (project/'ProjectSettings').mkdir(exist_ok=True)
    (project/'ProjectSettings/ProjectVersion.txt').write_bytes(
        (ROOT/'.venv/unity-candidate-check-2022/ProjectSettings/ProjectVersion.txt').read_bytes())
    template=Path(__file__).parent/'unity'
    runtime=(template/'EncoderScreenRuntime.cs.txt').read_text(encoding='utf8')
    runtime=runtime.replace('Unity.InferenceEngine','Unity.Sentis')
    runtime=runtime.replace('Worker worker = null','IWorker worker = null')
    runtime=runtime.replace('new Worker(model, BackendType.GPUCompute)','WorkerFactory.CreateWorker(BackendType.GPUCompute, model)')
    runtime=runtime.replace('worker.backendType.ToString()','((GenericWorker)worker).GetBackend().GetType().Name')
    runtime=runtime.replace('typeof(Worker)','typeof(GenericWorker)').replace('(HashSet<int>)fallbackField','(HashSet<string>)fallbackField')
    runtime=runtime.replace('fallback.Contains(l.outputs[0])','fallback.Contains(l.name)').replace('output = l.outputs[0]','output = l.name')
    runtime=runtime.replace('new Tensor<float>','new TensorFloat').replace('worker.Schedule(input)','worker.Execute(input)')
    runtime=runtime.replace('using (var output = ((Tensor<float>)worker.PeekOutput()).ReadbackAndClone())\n                    {',
                            '{ var output = (TensorFloat)worker.PeekOutput(); output.MakeReadable();')
    runtime=runtime.replace('using (var output = ((Tensor<float>)worker.PeekOutput()).ReadbackAndClone()) { output.DownloadToArray(); }',
                            'var output = (TensorFloat)worker.PeekOutput(); output.MakeReadable(); output.ToReadOnlyArray();')
    runtime=runtime.replace('output.DownloadToArray()','output.ToReadOnlyArray()')
    assert 'Tensor<float>' not in runtime and 'ReadbackAndClone' not in runtime
    (assets/'EncoderScreenRuntime.cs').write_text(runtime,encoding='utf8')
    (assets/'EncoderScreenMemory.cs').write_bytes((template/'EncoderScreenMemory.cs.txt').read_bytes())
    editor=(template/'EncoderScreenEditor.cs.txt').read_text(encoding='utf8').replace('using Unity.InferenceEngine;','using Unity.Sentis;')
    start=editor.index('            var assembly=')
    end=editor.index('            ModelWriter.Save',start)
    editor=editor[:start]+('            var converter = new Unity.Sentis.ONNX.ONNXModelConverter(true);\n'
                          '            var model = converter.Convert(source,Path.GetDirectoryName(source));\n')+editor[end:]
    (assets/'Editor/EncoderScreenEditor.cs').write_text(editor,encoding='utf8')
    print(project)


if __name__=='__main__': setup()
