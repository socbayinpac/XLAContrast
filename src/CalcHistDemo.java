import java.util.ArrayList;
import java.util.List;

import org.opencv.core.*;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

class CalcHist {
    public void run(String fileName) {

        Mat src = Imgcodecs.imread(fileName, Imgcodecs.IMREAD_GRAYSCALE);
        List<Mat> bgrPlanes = new ArrayList<>();
        bgrPlanes.add(src);

        int histSize = 256; // 255 level

        float[] range = {0, 256};
        MatOfFloat histRange = new MatOfFloat(range);

        //! Lay anh hist cua anh
        Mat hist = new Mat(), gHist = new Mat(), rHist = new Mat();
        Imgproc.calcHist(bgrPlanes, new MatOfInt(0), new Mat(), hist, new MatOfInt(histSize), histRange, false);

        //! Ve anh hist
        int histW = 512, histH = 400;
        int binW = (int) Math.round((double) histW / histSize);
        Mat histImage = new Mat( histH, histW, CvType.CV_8UC3, new Scalar( 0,0,0) );

        //! [Normalize the result to ( 0, histImage.rows )]
        Core.normalize(hist, hist, 0, histImage.rows(), Core.NORM_MINMAX);

        // lay data cua hist
        float[] histData = new float[(int)hist.total()];
        hist.get(0, 0, histData); // nap data vao mang

        for( int i = 1; i < histSize; i++ ) {
            Imgproc.line(histImage, new Point(binW * (i - 1), histH - Math.round(histData[i - 1])),
                    new Point(binW * (i), histH - Math.round(histData[i])), new Scalar(52, 96, 255), 2);
        }

        Imgcodecs.imwrite(fileName.substring(0,fileName.length()-4) +"-hist.jpg",histImage);// luu anh hist
    }
}

public class CalcHistDemo {
    public static String ORIGIN = "a.jpg";
    public static String RESULT = "result.jpg";
    public static String ORIGIN_HIST = "a-hist.jpg";
    public static String RESULT_HIST = "result-hist.jpg";


    public static void main(String[] args) {

        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        CalcHist calcHist = new CalcHist();
        histoEqual();
        calcHist.run(ORIGIN);
        calcHist.run(RESULT);
       showImages(new String[]{ ORIGIN,ORIGIN_HIST,RESULT,RESULT_HIST});
    }

    public static void showImages(String[] filename) {
        for (String name : filename) {
            Mat imread = Imgcodecs.imread(name);
            HighGui.imshow(name, imread);
        }
        HighGui.waitKey(0);
    }


    public static void histoEqual() {
        Mat imread = Imgcodecs.imread(ORIGIN, Imgcodecs.IMREAD_GRAYSCALE);
        Mat equ = new Mat();

        imread.copyTo(equ);

        // Equalizing the histogram of the image
        Imgproc.equalizeHist(equ, equ);

        Imgcodecs.imwrite(RESULT,equ);
        Imgcodecs.imwrite(ORIGIN,imread);

    }
}