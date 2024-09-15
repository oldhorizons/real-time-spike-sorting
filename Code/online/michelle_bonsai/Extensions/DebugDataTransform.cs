using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using Bonsai.Dsp;
using System.Linq;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Transform)]
public class DebugDataTransform
{
    public IObservable<SpikeWaveformCollection> Process(IObservable<SpikeWaveformCollection> source)
    {
        return source.Select(value => value);
    }
}
