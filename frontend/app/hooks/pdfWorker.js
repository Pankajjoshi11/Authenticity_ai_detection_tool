// src/pdfWorker.js
import * as pdfjsLib from "pdfjs-dist/build/pdf";
import pdfjsWorker from "pdfjs-dist/build/pdf.worker?worker";

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfjsWorker;
