const Loading = () => {
    return (
      <div className="flex justify-center items-center mt-4">
        <div className="w-8 h-8 border-4 border-blue-500 border-dashed rounded-full animate-spin"></div>
        <p className="ml-2">Processing...</p>
      </div>
    );
  };
  
  export default Loading;
  