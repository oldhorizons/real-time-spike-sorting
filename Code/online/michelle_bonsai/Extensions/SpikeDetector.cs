using Bonsai;
using System;
using System.ComponentModel;
using System.Collections.Generic;
using System.Linq;
using System.Reactive.Linq;
using OpenCV.Net;
using Bonsai.Reactive;
using Bonsai.Dsp;
using System.Collections.ObjectModel;
using System.Drawing.Printing;

[Combinator]
[Description("")]
[WorkflowElementCategory(ElementCategory.Combinator)]
public class SpikeDetector : Combinator<Mat, SpikeWaveformCollection>
{
    static readonly double[] DefaultThreshold = new[] { 0.0 };
    readonly Bonsai.Dsp.Delay delay = new Bonsai.Dsp.Delay();

    /// <summary>
    /// Gets or sets the delay of each spike waveform from its trigger, in samples.
    /// </summary>
    [Description("The delay of each spike waveform from its trigger, in samples.")]
    public int Delay
    {
        get { return delay.Count; }
        set { delay.Count = value; }
    }

    /// <summary>
    /// Gets or sets the length of each spike waveform, in samples.
    /// </summary>
    [Description("The length of each spike waveform, in samples.")]
    public int Length { get; set; }

    /// <summary>
    /// Gets or sets the per-channel threshold for detecting individual spikes.
    /// </summary>
    [TypeConverter(typeof(UnidimensionalArrayConverter))]
    [Editor("Bonsai.Dsp.Design.SpikeThresholdEditor, Bonsai.Dsp.Design", DesignTypes.UITypeEditor)]
    [Description("The per-channel threshold for detecting individual spikes.")]
    public double[] Threshold { get; set; }

    /// <summary>
    /// Gets or sets a value specifying the waveform refinement method.
    /// </summary>
    [Description("Specifies the waveform refinement method.")]
    public SpikeWaveformRefinement WaveformRefinement { get; set; }

    
    /// <summary>
    /// Detects spike events in the input signal and extracts their waveforms.
    /// </summary>
    /// <param name="source">
    /// A sequence of <see cref="Mat"/> objects representing the waveform of the
    /// signal from which to extract spike waveforms.
    /// </param>
    /// <returns>
    /// A sequence of <see cref="SpikeWaveformCollection"/> representing the spikes
    /// detected in each buffer of the signal waveform.
    /// </returns>
    public override IObservable<SpikeWaveformCollection> Process(IObservable<Mat> source)
    {
        return Observable.Defer(() =>
        {
            byte[] triggerBuffer = null;
            bool[] activeChannels = null;
            int[] refractoryChannels = null;
            SampleBuffer[] activeSpikes = null;
            long ioff = 0L;
            Console.WriteLine("01");
            return source.Publish(ps => ps.Zip(delay.Process(ps), (input, delayed) =>
            {
                SpikeWaveformCollection spikes = new SpikeWaveformCollection(input.Size);
                if (activeSpikes == null)
                {
                    Console.WriteLine("active spikes null");
                    triggerBuffer = new byte[input.Cols];
                    activeChannels = new bool[input.Rows];
                    refractoryChannels = new int[input.Rows];
                    activeSpikes = new SampleBuffer[input.Rows];
                }

                double[] thresholdValues = Threshold ?? DefaultThreshold;
                if (thresholdValues.Length == 0) thresholdValues = DefaultThreshold;
                Console.WriteLine("02");
                for (int i = 0; i < activeSpikes.Length; i++)
                {
                    Console.WriteLine("activespikes length");
                    using (Mat channel = input.GetRow(i))
                    using (Mat delayedChannel = delayed.GetRow(i))
                    {
                        Console.WriteLine("03");
                        double threshold = thresholdValues.Length > 1 ? thresholdValues[i] : thresholdValues[0];
                        if (activeSpikes[i] != null)
                        {
                            Console.WriteLine("activespikes not null");
                            SampleBuffer buffer = activeSpikes[i];
                            buffer = UpdateBuffer(buffer, delayedChannel, 0, delay.Count, threshold);
                            activeSpikes[i] = buffer;
                            if (buffer.Completed)
                            {
                                Console.WriteLine("buffer completed");
                                spikes.Add(new SpikeWaveform
                                {
                                    ChannelIndex = i,
                                    SampleIndex = buffer.SampleIndex,
                                    Waveform = buffer.Samples
                                });
                                activeSpikes[i] = null;
                            }
                            else continue;
                        }

                        using (Mat triggerHeader = Mat.CreateMatHeader(triggerBuffer))
                        {
                            Console.WriteLine("trigger header");
                            //TODO  - too much data lost?
                            Mat ch2 = new Mat(channel.Rows, channel.Cols, Depth.U8, 1);
                            CV.Normalize(channel, ch2, 0, 255, NormTypes.L2);
                            // CV.ConvertScale(channel, ch2, 0.00390625, 0);
                            
                            CV.Threshold(
                                ch2,
                                triggerHeader,
                                threshold, 1,
                                threshold < 0 ? ThresholdTypes.BinaryInv : ThresholdTypes.Binary);
                        }
                        Console.WriteLine("threshold finished");

                        for (int j = 0; j < triggerBuffer.Length; j++)
                        {
                            bool triggerHigh = triggerBuffer[j] > 0;
                            if (triggerHigh && !activeChannels[i] && refractoryChannels[i] == 0 && activeSpikes[i] == null)
                            {
                                Console.WriteLine("SPIKE DETECTED??");
                                int length = Length;
                                refractoryChannels[i] = length;
                                SampleBuffer buffer = new SampleBuffer(channel, length, j + ioff);
                                buffer.Refined |= WaveformRefinement == SpikeWaveformRefinement.None;
                                buffer = UpdateBuffer(buffer, delayedChannel, j, delay.Count, threshold);
                                if (buffer.Completed)
                                {
                                    Console.WriteLine("buffer completed");
                                    spikes.Add(new SpikeWaveform
                                    {
                                        ChannelIndex = i,
                                        SampleIndex = buffer.SampleIndex,
                                        Waveform = buffer.Samples
                                    });
                                }
                                else activeSpikes[i] = buffer;
                                Console.WriteLine("buffer goin");
                            }

                            activeChannels[i] = triggerHigh;
                            if (refractoryChannels[i] > 0)
                            {
                                Console.WriteLine("channel refractory");
                                refractoryChannels[i]--;
                            }
                        }
                    }
                }

                ioff += input.Cols;
                return spikes;  
            }));
        });
    }

    static SampleBuffer UpdateBuffer(SampleBuffer buffer, Mat source, int index, int delay, double threshold)
    {
        Console.WriteLine("started update buffer");
        int samplesTaken = buffer.Update(source, index);
        if (buffer.Completed && !buffer.Refined)
        {
            Mat waveform = buffer.Samples;
            double minVal, maxVal;
            Point minLoc, maxLoc;
            CV.MinMaxLoc(waveform, out minVal, out maxVal, out minLoc, out maxLoc);
            int offset = threshold > 0 ? maxLoc.X - delay : minLoc.X - delay;
            if (offset > 0)
            {
                SampleBuffer offsetBuffer = new SampleBuffer(waveform, waveform.Cols, buffer.SampleIndex + offset);
                offsetBuffer.Refined = true;
                offsetBuffer.Update(waveform, offset);
                offsetBuffer.Update(source, index + samplesTaken + offset);
                return offsetBuffer;
            }
        }
        Console.WriteLine("updated buffer");

        return buffer;
    }
}

class SampleBuffer
{
    int offset;
    readonly long sampleIndex;
    readonly Mat samples;

    public SampleBuffer(Mat template, int count, long index)
    {
        sampleIndex = index;
        if (count > 0)
        {
            samples = new Mat(template.Rows, count, template.Depth, template.Channels);
        }
        else Refined = true;
    }

    public bool Refined { get; set; }

    public long SampleIndex
    {
        get { return sampleIndex; }
    }

    public Mat Samples
    {
        get { return samples; }
    }

    public bool Completed
    {
        get { return samples == null || offset >= samples.Cols; }
    }

    public int Update(Mat source, int index)
    {
        int windowElements;
        if (samples != null && (windowElements = Math.Min(source.Cols - index, samples.Cols - offset)) > 0)
        {
            using (var dataSubRect = source.GetSubRect(new Rect(index, 0, windowElements, source.Rows)))
            using (var windowSubRect = samples.GetSubRect(new Rect(offset, 0, windowElements, samples.Rows)))
            {
                CV.Copy(dataSubRect, windowSubRect);
            }

            offset += windowElements;
            return windowElements;
        }

        return 0;
    }
}
