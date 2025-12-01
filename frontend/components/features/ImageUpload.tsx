'use client';

import { useState, useRef, ChangeEvent, DragEvent } from 'react';
import { useTranslation } from 'react-i18next';
import { FiUpload, FiCamera, FiX, FiImage } from 'react-icons/fi';

interface ImageUploadProps {
  onImageSelect: (file: File) => void;
  onClear?: () => void;
  isLoading?: boolean;
  maxSizeMB?: number;
}

export default function ImageUpload({
  onImageSelect,
  onClear,
  isLoading = false,
  maxSizeMB = 10,
}: ImageUploadProps) {
  const { t } = useTranslation();
  const [preview, setPreview] = useState<string | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const cameraInputRef = useRef<HTMLInputElement>(null);

  // Validate file
  const validateFile = (file: File): boolean => {
    setError(null);

    // Check file type
    if (!file.type.startsWith('image/')) {
      setError(t('invalidImage'));
      return false;
    }

    // Check file size
    const maxSize = maxSizeMB * 1024 * 1024; // Convert MB to bytes
    if (file.size > maxSize) {
      setError(`File size must be less than ${maxSizeMB}MB`);
      return false;
    }

    return true;
  };

  // Handle file selection
  const handleFileSelect = (file: File) => {
    if (!validateFile(file)) return;

    // Create preview
    const reader = new FileReader();
    reader.onload = (e) => {
      setPreview(e.target?.result as string);
    };
    reader.readAsDataURL(file);

    // Pass to parent
    onImageSelect(file);
  };

  // Handle file input change
  const handleFileChange = (e: ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // Handle drag events
  const handleDragEnter = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
  };

  const handleDrop = (e: DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);

    const file = e.dataTransfer.files?.[0];
    if (file) {
      handleFileSelect(file);
    }
  };

  // Handle clear
  const handleClear = () => {
    setPreview(null);
    setError(null);
    if (fileInputRef.current) fileInputRef.current.value = '';
    if (cameraInputRef.current) cameraInputRef.current.value = '';
    onClear?.();
  };

  // Open file picker
  const openFilePicker = () => {
    fileInputRef.current?.click();
  };

  // Open camera
  const openCamera = () => {
    cameraInputRef.current?.click();
  };

  return (
    <div className="w-full">
      {/* Hidden inputs */}
      <input
        ref={fileInputRef}
        type="file"
        accept="image/*"
        onChange={handleFileChange}
        className="hidden"
      />
      <input
        ref={cameraInputRef}
        type="file"
        accept="image/*"
        capture="environment"
        onChange={handleFileChange}
        className="hidden"
      />

      {/* Upload Area */}
      {!preview ? (
        <div
          onDragEnter={handleDragEnter}
          onDragLeave={handleDragLeave}
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          className={`
            border-4 border-dashed rounded-xl p-12 text-center
            transition-all duration-300 cursor-pointer
            ${
              isDragging
                ? 'border-primary-500 bg-primary-50'
                : 'border-gray-300 hover:border-primary-400 bg-white'
            }
            ${isLoading ? 'opacity-50 pointer-events-none' : ''}
          `}
          onClick={openFilePicker}
        >
          <div className="flex flex-col items-center gap-4">
            {/* Icon */}
            <div
              className={`
              text-6xl transition-transform duration-300
              ${isDragging ? 'scale-110' : 'scale-100'}
            `}
            >
              {isDragging ? '📂' : '📸'}
            </div>

            {/* Title */}
            <h3 className="text-2xl font-semibold text-gray-800">
              {t('uploadImage')}
            </h3>

            {/* Subtitle */}
            <p className="text-gray-600">{t('dragDrop')}</p>

            {/* Divider */}
            <p className="text-sm text-gray-500">{t('or')}</p>

            {/* Buttons */}
            <div className="flex gap-4 flex-wrap justify-center">
              <button
                onClick={(e) => {
                  e.stopPropagation();
                  openFilePicker();
                }}
                disabled={isLoading}
                className="btn btn-primary text-lg px-6 py-3 flex items-center gap-2"
              >
                <FiUpload className="text-xl" />
                {t('uploadImage')}
              </button>

              <button
                onClick={(e) => {
                  e.stopPropagation();
                  openCamera();
                }}
                disabled={isLoading}
                className="btn btn-secondary text-lg px-6 py-3 flex items-center gap-2"
              >
                <FiCamera className="text-xl" />
                {t('takePhoto')}
              </button>
            </div>

            {/* File info */}
            <p className="text-xs text-gray-400 mt-2">
              JPG, PNG, WEBP • Max {maxSizeMB}MB
            </p>
          </div>
        </div>
      ) : (
        /* Preview Area */
        <div className="card p-6">
          <div className="relative">
            {/* Clear button */}
            {!isLoading && (
              <button
                onClick={handleClear}
                className="absolute top-2 right-2 z-10 bg-red-500 hover:bg-red-600 text-white rounded-full p-2 transition-all duration-200 shadow-lg hover:scale-110"
                title="Remove image"
              >
                <FiX className="text-xl" />
              </button>
            )}

            {/* Image preview */}
            <div className="relative overflow-hidden rounded-lg">
              <img
                src={preview}
                alt="Preview"
                className={`
                  w-full h-auto max-h-96 object-contain
                  ${isLoading ? 'opacity-50' : ''}
                `}
              />

              {/* Loading overlay */}
              {isLoading && (
                <div className="absolute inset-0 flex items-center justify-center bg-black bg-opacity-30">
                  <div className="text-center text-white">
                    <div className="animate-spin text-5xl mb-3">⏳</div>
                    <p className="text-lg font-semibold">
                      {t('recognizing')}
                    </p>
                    <p className="text-sm">{t('pleaseWait')}</p>
                  </div>
                </div>
              )}
            </div>

            {/* Image info */}
            {!isLoading && (
              <div className="mt-4 flex items-center justify-between text-sm text-gray-600">
                <div className="flex items-center gap-2">
                  <FiImage />
                  <span>Image ready</span>
                </div>
                <button
                  onClick={openFilePicker}
                  className="text-primary-600 hover:text-primary-700 font-medium"
                >
                  {t('tryAnother')}
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Error message */}
      {error && (
        <div className="mt-4 p-4 bg-red-50 border-2 border-red-200 rounded-lg text-red-700">
          <p className="font-medium">⚠️ {t('error')}</p>
          <p className="text-sm mt-1">{error}</p>
        </div>
      )}
    </div>
  );
}
