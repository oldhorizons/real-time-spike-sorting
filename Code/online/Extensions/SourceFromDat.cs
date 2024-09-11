using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;

[Combinator]
[Description("produces a sequence of Rhd2164DataFrame objects from a .dat file")]
[WorkflowElementCategory(ElementCategory.Source)]
public class SourceFromDat : Source<Rhd2164DataFrame>
{
    [TypeConverter(typeof(NameConverter))]
    [Description("The unique device name.")]
    [Category("Configuration")]
    public string FilePath { get; set }

    [Description("The number of samples collected for each channel that are used to create a single Rhd2164DataFrame.")]
    [Category("Configuration")]
    public int BufferSize { get; set; }

    private int FilePosition { get; set; } = 0;

    public override IObservable<Rhd2164DataFrame> Generate()
    {
        using (BinaryReader b = new BinaryReader(File.Open(FilePath, FileMode.Open)))
        { 
            int pos = 0;
            int length = (int)
        }
        return new Rhd2164DataFrame
        {
            AmplifierData = Mat
            AuxData = Mat
            Clock = ulong
            HubClock = ulong
        };
    }
    public IObservable<int> Process()
    {
        return Observable.Return(0);
    }
    // public override bool Equals(object obj);
    // public override int GetHashCode();
    // public Type GetType();
    // public override string ToString();
}



