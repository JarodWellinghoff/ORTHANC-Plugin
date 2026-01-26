import cornerstone from "cornerstone-core";
import cornerstoneWADOImageLoader from "cornerstone-wado-image-loader";
import dicomParser from "dicom-parser";
import cornerstoneTools from "cornerstone-tools";
import cornerstoneMath from "cornerstone-math";
import Hammer from "hammerjs";

const workerConfig = {
  webWorkerPath: new URL(
    /* @vite-ignore */ "cornerstone-wado-image-loader/dist/cornerstoneWADOImageLoaderWebWorker.js",
    import.meta.url
  ).href,
  taskConfiguration: {
    decodeTask: {
      codecsPath: new URL(
        /* @vite-ignore */ "cornerstone-wado-image-loader/dist/cornerstoneWADOImageLoaderCodecs.js",
        import.meta.url
      ).href,
    },
  },
};

let cornerstoneReady = false;
let toolsReady = false;

export const API_BASE_URL =
  import.meta.env.VITE_API_URL?.replace(/\/$/, "") ?? "";

export const ensureCornerstoneReady = () => {
  if (cornerstoneReady || typeof window === "undefined") {
    return;
  }

  cornerstoneWADOImageLoader.external.cornerstone = cornerstone;
  cornerstoneWADOImageLoader.external.dicomParser = dicomParser;
  cornerstoneWADOImageLoader.configure({ useWebWorkers: true });

  const maxWorkers = Math.min(window.navigator?.hardwareConcurrency ?? 2, 4);

  cornerstoneWADOImageLoader.webWorkerManager.initialize({
    webWorkerPath: workerConfig.webWorkerPath,
    taskConfiguration: workerConfig.taskConfiguration,
    maxWebWorkers: maxWorkers,
    startWebWorkersOnDemand: true,
  });

  cornerstoneReady = true;
};

export const ensureCornerstoneToolsReady = () => {
  if (toolsReady || typeof window === "undefined") {
    return;
  }

  ensureCornerstoneReady();

  cornerstoneTools.enableLogger();
  localStorage.setItem("debug", "cornerstoneTools:*");

  cornerstoneTools.external.cornerstone = cornerstone;
  cornerstoneTools.external.cornerstoneMath = cornerstoneMath;
  cornerstoneTools.external.Hammer = Hammer;
  cornerstoneTools.external.dicomParser = dicomParser;

  cornerstoneTools.init({
    showSVGCursors: true,
    globalToolSyncEnabled: true,
    mouseEnabled: true,
    touchEnabled: true,
  });

  [
    cornerstoneTools.WwwcTool,
    cornerstoneTools.PanTool,
    cornerstoneTools.ZoomTool,
    cornerstoneTools.StackScrollMouseWheelTool,
  ].forEach((ToolClass) => {
    if (ToolClass) {
      cornerstoneTools.addTool(ToolClass);
    }
  });

  toolsReady = true;
};

export const cornerstoneApi = {
  cornerstone,
  cornerstoneTools,
  cornerstoneWADOImageLoader,
};
